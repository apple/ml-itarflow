#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
__all__ = [
    'CosineLRSchedule',
    'Distributed',
    'FID',
    'Metrics',
    'get_data',
    'set_random_seed',
]

import datetime
import hashlib
import math
import os
import pathlib
import random
from collections import Counter

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
import torch.utils.data
import torchvision as tv
import torchvision.transforms.functional as TF

from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from torchvision.transforms import InterpolationMode as I


class CosineLRSchedule(torch.nn.Module):
    counter: torch.Tensor

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
        super().__init__()
        self.register_buffer('counter', torch.zeros(()))
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.set_lr(min_lr)

    def set_lr(self, lr: float) -> float:
        if self.min_lr <= lr <= self.max_lr:
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        return max(self.min_lr, min(self.max_lr, lr))

    def step(self) -> float:
        with torch.no_grad():
            counter = self.counter.add_(1).item()
        if self.counter <= self.warmup_steps:
            new_lr = self.min_lr + counter / self.warmup_steps * (self.max_lr - self.min_lr)
            return self.set_lr(new_lr)

        t = (counter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        new_lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (self.max_lr - self.min_lr)
        return self.set_lr(new_lr)


class Distributed:
    timeout: float = 72000

    def __init__(self):
        if os.environ.get('MASTER_PORT'):  # When running with torchrun
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = True
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                timeout=datetime.timedelta(seconds=self.timeout),
                rank=self.rank,
            )
        else:  # When running with python for debugging
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.distributed = False
        torch.cuda.set_device(self.local_rank)
        self.barrier()

    def barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()

    def gather_concat(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x
        x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.cat(x_list)

    def reduce(self, x):
        if not self.distributed:
            return x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        return x

    def __del__(self):
        if self.distributed:
            torch.distributed.destroy_process_group()


class FID(FrechetInceptionDistance):
    def __init__(self, feature=2048, reset_real_features=True, normalize=False, input_img_size=..., **kwargs):
        super().__init__(feature, reset_real_features, normalize, input_img_size, **kwargs)
        self.reset_real_features = reset_real_features

    def add_state(self, name, default, *args, **kwargs):
        self.register_buffer(name, default)

    def manual_compute(self, dist):
        self.fake_features_num_samples = dist.reduce(self.fake_features_num_samples)
        self.fake_features_sum = dist.reduce(self.fake_features_sum)
        self.fake_features_cov_sum = dist.reduce(self.fake_features_cov_sum)

        if self.reset_real_features:
            self.real_features_num_samples = dist.reduce(self.real_features_num_samples)
            self.real_features_sum = dist.reduce(self.real_features_sum)
            self.real_features_cov_sum = dist.reduce(self.real_features_cov_sum)

        print(f'Gathered {self.fake_features_num_samples} samples for FID computation')

        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        if dist.rank == 0:
            fid_score = _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(
                dtype=self.orig_dtype, device=self.real_features_sum.device)
            print(f'FID: {fid_score.item()} DONE')
        else:
            fid_score = torch.tensor(0.0, dtype=self.orig_dtype, device=self.real_features_sum.device)
        dist.barrier()

        self.fake_features_num_samples *= 0
        self.fake_features_sum *= 0
        self.fake_features_cov_sum *= 0

        if self.reset_real_features:
            self.real_features_num_samples *= 0
            self.real_features_sum *= 0
            self.real_features_cov_sum *= 0

        return fid_score


class Metrics:
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, torch.Tensor | float]):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

    def compute(self, dist: Distributed | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.metrics.items():
            v = sum(v) / len(v)
            if dist is not None:
                v = dist.gather_concat(torch.tensor(v, device='cuda').view(1)).mean().item()
            out[k] = v
        return out

    @staticmethod
    def print(metrics: dict[str, float], epoch: int):
        print(f'Epoch {epoch}  Time {datetime.datetime.now()}')
        print('\n'.join((f'\t{k:40s}: {v: .4g}' for k, v in sorted(metrics.items()))))


def get_num_classes(dataset: str) -> int:
    return {'cifar10': 10, 'imagenet64': 0, 'imagenet': 1000, 'food101': 101, 'afhq': 3, 'ffhq256': 0, 'edm_imagenet': 1000, 'adm_imagenet': 1000, 'imagenet_eval': 1000}[dataset]


class EDMPreprocess:
    """Center-crop to the largest square, then Lanczos resize to (S,S)."""
    def __init__(self, size: int):
        self.size = size
    def __call__(self, im):
        im = im.convert("RGB")
        s = min(im.size)
        im = TF.center_crop(im, s)
        im = TF.resize(im, (self.size, self.size), interpolation=I.LANCZOS, antialias=True)
        return im


class ADMPreprocess:
    """Dhariwal/ADM: progressive BOX downsample by 2x, Bicubic to short_side=S, then center crop SxS."""
    def __init__(self, size: int):
        self.size = size
    def __call__(self, im):
        im = im.convert("RGB")
        while min(im.size) >= 2 * self.size:
            new_w, new_h = im.size[0] // 2, im.size[1] // 2
            im = TF.resize(im, (new_h, new_w), interpolation=I.BOX, antialias=True)
        w, h = im.size
        scale = self.size / min(w, h)
        new_size = (round(w * scale), round(h * scale))
        im = TF.resize(im, new_size, interpolation=I.BICUBIC, antialias=True)
        im = TF.center_crop(im, self.size)
        return im


def get_data(dataset: str, img_size: int, folder: pathlib.Path, xflip: int) -> tuple[torch.utils.data.Dataset, int]:
    if 'edm' in dataset or 'adm' in dataset:
        preprocessing_name = 'edm' if 'edm' in dataset else 'adm'
        print(f'using {preprocessing_name} preprocessing')

        transform_list = [EDMPreprocess(img_size) if preprocessing_name == 'edm' else ADMPreprocess(img_size)]

        if xflip: transform_list.append(tv.transforms.RandomHorizontalFlip())

        transform_list.extend([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform = tv.transforms.Compose(transform_list)

    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(img_size),
                tv.transforms.CenterCrop(img_size),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    if dataset == 'cifar10':
        data = tv.datasets.CIFAR10(folder, transform=transform, train=True, download=False)
    elif dataset == 'imagenet64':
        data = tv.datasets.ImageFolder(str(folder / 'imagenet64/train'), transform=transform)
    elif dataset == 'imagenet':
        data = tv.datasets.ImageFolder(str(folder / 'imagenet/train'), transform=transform)
    elif dataset == 'imagenet_eval':
        print('-------warning: using Evaluation dataset --------------')
        data = tv.datasets.ImageFolder(str(folder / 'imagenet/val'), transform=transform)
    elif dataset == 'edm_imagenet':
        data = tv.datasets.ImageFolder(str(folder / 'imagenet/train'), transform=transform)
    elif dataset == 'adm_imagenet':
        data = tv.datasets.ImageFolder(str(folder / 'imagenet/train'), transform=transform)
    elif dataset == 'food101':
        data = tv.datasets.ImageFolder(str(folder / 'food101'), transform=transform)
    elif dataset == 'afhq':
        data = tv.datasets.ImageFolder(str(folder / 'afhq'), transform=transform)
    elif dataset == 'ffhq256':
        data = tv.datasets.ImageFolder(str(folder / 'ffhq256'), transform=transform)
    else:
        raise NotImplementedError(f'Unknown dataset {dataset}')
    return data, get_num_classes(dataset)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseWriter(object):
    def __init__(self, opt):
        pass
    def add_scalar(self, step, key, val):
        pass
    def add_image(self, step, key, image):
        pass
    def add_dict(self, step, dict):
        pass
    def close(self):
        pass


class PrintWriter(BaseWriter):
    def __init__(self, opts):
        super(PrintWriter, self).__init__(opts)
        print(f'[PrintWriter] Initialized for experiment: {opts.exp_name}')

    def add_scalar(self, step, key, val):
        print(f'[step {step}] {key}: {val}')

    def add_dict(self, step, dict):
        for key, val in dict.items():
            print(f'[step {step}] {key}: {val}')

    def add_image(self, step, key, image):
        print(f'[step {step}] {key}: image logged (shape={image.shape})')


@torch.no_grad()
def ddp_batch_sanity_check(x, ids=None, tag="train", strict=True):
    """
    x:   Tensor on this rank, shape [B, ...]
    ids: Optional iterable of length B with stable sample IDs
         (e.g., dataset indices or file paths). Strongly recommended.
         If None, falls back to content hashes (slow; use occasionally).
    strict: raise RuntimeError on failures if True, else just print warnings.

    Returns: dict with {"global_bs", "expected", "ok_batch"}
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    rank  = dist.get_rank()
    world = dist.get_world_size()
    device = x.device

    local_bs = int(x.size(0))
    t = torch.tensor([local_bs], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    global_bs = int(t.item())
    expected  = local_bs * world
    ok_batch  = (global_bs == expected)

    if rank == 0:
        print(f"[{tag}] local_bs={local_bs}, world={world}, global_bs={global_bs}, expected={expected}")
        if not ok_batch:
            msg = f"[{tag}] WARNING: global_bs({global_bs}) != local_bs*world({expected})."
            print(msg)
            if strict:
                raise RuntimeError(msg)

    if ids is None:
        xs = x.detach().contiguous().cpu()
        local_keys = []
        for b in range(xs.size(0)):
            digest = hashlib.sha1(xs[b].numpy().tobytes()).digest()
            local_keys.append(int.from_bytes(digest[:8], "little"))
    else:
        if torch.is_tensor(ids):
            ids = ids.view(-1).cpu().tolist()
        local_keys = []
        for v in ids:
            if isinstance(v, (int,)):
                local_keys.append(int(v))
            elif isinstance(v, str):
                digest = hashlib.sha1(v.encode("utf-8")).digest()
                local_keys.append(int.from_bytes(digest[:8], "little"))
            else:
                digest = hashlib.sha1(repr(v).encode("utf-8")).digest()
                local_keys.append(int.from_bytes(digest[:8], "little"))

    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_keys)

    if rank == 0:
        flat = [k for part in gathered for k in part]
        cnt  = Counter(flat)
        dups = [k for k, c in cnt.items() if c > 1]
        if dups:
            msg = f"[{tag}] DUPLICATE samples across ranks: {len(dups)} duplicates (showing up to 10): {dups[:10]}"
            print(msg)
            if strict:
                raise RuntimeError(msg)
        else:
            print(f"[{tag}] OK: batch keys are unique across {world} ranks (total {len(flat)} samples).")

    return {"global_bs": global_bs, "expected": expected, "ok_batch": ok_batch}
