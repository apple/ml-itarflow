#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import argparse
import builtins
import pathlib

import numpy as np
import torch
import torch.utils.data
import torchvision as tv
from pathlib import Path
import PIL.Image

import transformer_flow
import utils
import os


def self_denoising(model,y,samples,args,dist,t):
    samples = samples.cpu()
    assert args.sample_batch_size % args.denoising_batch_size == 0
    db = args.denoising_batch_size // dist.world_size
    base_lr = db * args.img_size**2 * args.channel_size * t**2
    lr = base_lr
    denoised_samples = []
    for j in range(args.sample_batch_size // args.denoising_batch_size):
        x = torch.clone(samples[j * db : (j + 1) * db]).detach().cuda()
        tt = torch.full((x.shape[0], 1), t, device=x.device)
        x.requires_grad = True
        y_ = y[j * db : (j + 1) * db] if y is not None else None
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            z, _, logdets,_ = model(x, y_,t=tt)
        loss =(0.5 * z.pow(2).mean(dim=[1, 2]) - logdets).mean()
        grad = torch.autograd.grad(loss, [x], retain_graph=False, create_graph=False)[0]
        x.data.add_(grad, alpha=-lr)
        denoised_samples.append(x.detach().cpu())
    samples = torch.cat(denoised_samples, dim=0).cuda()
    return samples

def iterative_denoising(model,y,samples,args,dist,t0=1e-2,t1=3e-1,steps=None):
    samples = samples.cpu()
    steps   = int(t1/1e-2) if steps is None else steps
    ts_base = torch.linspace(t1, t0, steps, device=samples.device)
    ts      = torch.cat([ts_base, ts_base.new_tensor([0.0])])
    dts     = ts[:-1] - ts[1:]

    x=samples
    for i, tt in enumerate(ts[:-1]):
        x = x.detach().cuda()
        x0 = self_denoising(model,y,x,args,dist,tt)
        if args.clip: x0=x0.clamp(min=-1,max=1)
        dt = dts[i]
        with torch.no_grad():
            v  = (x-x0) / tt
            if i!= ts[:-1].shape[0]-1:
                if args.dyn=='sde_vanilla':
                    alpha   = (ts[i+1]/ts[i])**2
                    v       = alpha**0.5 * v + (1 - alpha)**0.5 *torch.randn_like(v)
                    x       = x0 + ts[i+1]*v

                elif args.dyn=='ode':
                    x  = x - v * dt

                else:
                    raise RuntimeError
            else:
                x  = x - v * dt
        if x.isnan().any().item():
            print(f'find NaN at Iterative Sampling at t={tt}')
            return x
        if x.isinf().any().item():
            print(f'find inf at Iterative Sampling at t={tt}')
            return x
    return x


def main(args):
    dist = utils.Distributed()
    utils.set_random_seed(100 + dist.rank)

    def print(*args, **kwargs):
        if dist.local_rank == 0:
            builtins.print(*args, **kwargs)

    num_classes = utils.get_num_classes(args.dataset)

    fid_stats_file = args.data / f'{args.dataset}_{args.img_size}_fid_stats.pth'
    print(f'evaluating {fid_stats_file}')
    assert fid_stats_file.exists()
    print(f'Loading FID stats from {fid_stats_file}')
    fid = utils.FID(reset_real_features=False, normalize=True).cuda()
    fid.load_state_dict(torch.load(fid_stats_file, map_location='cpu', weights_only=True))
    dist.barrier()

    model = transformer_flow.Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_sizes=args.patch_sizes,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        num_classes=num_classes,
        input_scale=args.input_scale,
        shared_backbone_layers=args.shared_backbone_layers,
    ).cuda()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    model_name = Path(args.ckpt_file).stem
    sample_dir: pathlib.Path = args.logdir / f'{args.dataset}_samples_{model_name}'

    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt_file, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    dist.barrier()

    num_batches = int(np.ceil(args.num_samples / args.sample_batch_size))
    last_batch_size = args.num_samples - (num_batches - 1) * args.sample_batch_size

    def get_noise(b):
        return torch.randn(
            b, (args.img_size // args.patch_sizes[-1]) ** 2, args.channel_size * args.patch_sizes[-1]**2, device='cuda'
        )

    os.makedirs(args.img_dir, exist_ok=True)
    global_sample_idx = 0

    for i in range(num_batches):
        noise = get_noise(args.sample_batch_size // dist.world_size).to('cuda')
        if num_classes:
            y = torch.randint(num_classes, (args.sample_batch_size // dist.world_size,), device='cuda')
        else:
            y = None

        while True:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32):
                init_t = args.t1
                t = torch.ones(noise.shape[0], 1, device=noise.device) * init_t
                samples = model.reverse(noise, y, args.cfg, attn_temp=args.attn_temp,
                                        annealed_guidance=args.annealed_guidance, t=t, lerp_style=args.lerp_style,guide_flows=args.guide_flows)

                if samples.isnan().any().item():
                    print('find NaN at TarFlow Sampling')
                if samples.isinf().any().item():
                    print('find Inf at TarFlow Sampling')

            samples = iterative_denoising(model, y, samples, args, dist, t0=args.t0, t1=args.t1, steps=args.itr_steps)

            samples = dist.gather_concat(samples.detach())

            if not (samples.isnan().any().item() or samples.isinf().any().item()):
                break
            else:
                print('find NaN or inf, retrying...')
                noise = get_noise(args.sample_batch_size // dist.world_size).to('cuda')

        if i == num_batches - 1:
            samples = samples[:last_batch_size]

        if not args.edm_eval:
            samples = 0.5 * (samples.clip(-1, 1) + 1)
        else:
            samples = 0.5 * (samples + 1)

        if dist.rank == 0:
            b, c, h, w = samples.shape
            if not args.edm_eval:
                samples_uint8 = (samples * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            else:
                samples_uint8 = (samples * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

            for j in range(b):
                filename = f"{global_sample_idx:06d}.png"
                filepath = os.path.join(sample_dir, filename)
                PIL.Image.fromarray(samples_uint8[j], 'RGB').save(filepath)
                global_sample_idx += 1

        dist.barrier()
        fid.update(samples, real=False)
        print(f'{i}/{num_batches} batch sample complete')

    fid_score = fid.compute().item()
    fid.reset()

    print(f'{args.ckpt_file} {model_name} cfg {args.cfg:.2f} fid {fid_score:.2f}')
    if dist.local_rank == 0:
        # Save a grid of the last batch for visual inspection
        tv.utils.save_image(samples.clip(min=-1, max=1), sample_dir / f'samples_cfg{args.cfg:.2f}.png', normalize=False, nrow=16)
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', type=pathlib.Path)
    parser.add_argument('--logdir', default='runs', type=pathlib.Path)

    parser.add_argument('--ckpt_file', default='', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'food101', 'imagenet', 'imagenet64', 'afhq', 'ffhq256'])
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--channel_size', default=3, type=int)
    parser.add_argument('--patch_sizes', default=[4], type=int, nargs='+')
    parser.add_argument('--blocks', default=4, type=int)
    parser.add_argument('--channels', default=[512], type=int, nargs='+')
    parser.add_argument('--layers_per_block', default=[8], type=int, nargs='+')
    parser.add_argument('--nvp', default=1, type=int)
    parser.add_argument('--annealed_guidance', default=1, type=int)
    parser.add_argument('--input_scale', default=1, type=float)

    parser.add_argument('--cfg', default=0, type=float)
    parser.add_argument('--attn_temp', default=1, type=float)
    parser.add_argument('--num_samples', default=50000, type=int)
    parser.add_argument('--sample_batch_size', default=1024, type=int)
    parser.add_argument('--denoising_batch_size', default=256, type=int)
    parser.add_argument('--img_dir', default='out', type=str)
    parser.add_argument('--edm_eval', default=0, type=int)
    parser.add_argument('--clip', default=0, type=int)
    parser.add_argument('--itr_steps', default=None, type=int)
    parser.add_argument('--dyn', default='ode', type=str)
    parser.add_argument('--lerp_style', default='tarflow', type=str)
    parser.add_argument('--t0', default=1e-2, type=float)
    parser.add_argument('--t1', default=3e-1, type=float)
    parser.add_argument('--dtype', default='bfloat16', type=str)
    parser.add_argument('--ts', default=[5e-2], type=float, nargs='+')
    parser.add_argument('--guide_flows', default=-1, type=int)
    parser.add_argument('--shared_backbone_layers', default=0, type=int)
    args = parser.parse_args()

    main(args)
