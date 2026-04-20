#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import argparse
import builtins
import pathlib
import os

import torch
import torch.amp
import torch.utils
import torch.utils.data
import torchvision as tv
from utils import PrintWriter, BaseWriter
from evaluate_fid import iterative_denoising

import transformer_flow
import utils
import pickle
import gc

def sample_uniform(t0, t1, N, device, dtype):
    return torch.rand(N, device=device, dtype=dtype) * (t1 - t0) + t0

def main(args):
    dist = utils.Distributed()
    utils.set_random_seed(100 + dist.rank)
    data, num_classes = utils.get_data(args.dataset, args.img_size, args.data,xflip=args.xflip)

    def print(*args, **kwargs):
        if dist.local_rank == 0:
            builtins.print(*args, **kwargs)

    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:32s}: {v}')

    args.logdir.mkdir(parents=True, exist_ok=True)
    with open(args.logdir / "config.pkl", "wb") as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('---------config saved------------')

    if dist.rank == 0:
        writer = BaseWriter(args) if args.debug else PrintWriter(args)

    fid = utils.FID(reset_real_features=False, normalize=True, sync_on_compute=False).to('cuda')
    fid_stats_file = args.data / f'{args.dataset}_{args.img_size}_fid_stats.pth'
    if fid_stats_file.exists():
        print(f'Loading FID stats from {fid_stats_file}')
        fid.load_state_dict(torch.load(fid_stats_file, map_location='cpu', weights_only=True))
    else:
        raise FileNotFoundError(f'FID stats file "{fid_stats_file}" not found, run prepare_fid_stats.py.')
    dist.barrier()

    fixed_noise = torch.randn(
        args.num_samples // dist.world_size,
        (args.img_size // args.patch_sizes[-1]) ** 2,
        args.channel_size * args.patch_sizes[-1]**2,
    )
    if num_classes:
        fixed_y = torch.randint(num_classes, (args.num_samples // dist.world_size,))
    else:
        fixed_y = None
    data_sampler = torch.utils.data.DistributedSampler(data, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        data,
        sampler=data_sampler,
        batch_size=args.batch_size // dist.world_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

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
        attn_dropout=args.attn_dropout,
        shared_backbone_layers=args.shared_backbone_layers,
    ).to('cuda')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameter is {trainable}')

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    lr_schedule = utils.CosineLRSchedule(optimizer, len(data_loader), args.epochs * len(data_loader), 1e-5, args.lr)

    if args.ckpt_file is not None:
        ckpt = torch.load(args.ckpt_file, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt)
        del ckpt
        print(f'Loaded checkpoint {args.ckpt_file}')

    if dist.distributed:
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank])
    else:
        model_ddp = model

    if args.loss_scaling:
        scaler = torch.amp.GradScaler()
    model_name = f'{args.dataset}_{args.patch_sizes}_{args.channels}_{args.blocks}_{args.layers_per_block}'
    sample_dir: pathlib.Path = args.logdir / f'{args.dataset}_samples_{model_name}'
    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(x, y,t,_t,fp16):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if fp16 else torch.float32):
            z, outputs, logdets, mid_vars = model_ddp(x, y, t)
            loss = model.get_loss(z, logdets,mid_vars,_t)
            return loss, (z, outputs, logdets,mid_vars)

    if args.compile:
        compute_loss = torch.compile(compute_loss, fullgraph=False, backend='inductor', mode='max-autotune')
        dist.barrier()

    itr=0
    print(f'{" Training ":-^80}')
    for epoch in range(args.epochs):
        metrics = utils.Metrics()
        for ii,(x, y) in enumerate(data_loader):
            x = x.cuda()

            if epoch ==0 and ii==0:
                out = utils.ddp_batch_sanity_check(x, ids=None, tag="train", strict=True)
                print('-----------passed dist dataloader unit test------------')

            if args.t_sample=='uni':
                t               = sample_uniform(args.t0,args.t1,x.shape[0], x.device,x.dtype)[:,None]
                eps             = t.reshape(x.shape[0],1,1,1) * torch.randn_like(x)
                nn_input_t      = t
            elif args.t_sample=='edm':
                P_std=1.0
                P_mean=-1.2
                rnd_normal = torch.randn([x.shape[0], 1], device=x.device)
                t = (rnd_normal * P_std + P_mean).exp()
                eps = t.reshape(x.shape[0],1,1,1)*torch.randn_like(x)
                nn_input_t      = t
            elif args.t_sample== 'linear':
                u = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
                t = torch.sqrt(u * (args.t1 ** 2 - args.t0 ** 2) + args.t0 ** 2)[:, None]
                eps = t.view(x.shape[0], 1, 1, 1) * torch.randn_like(x)
                nn_input_t = t
            else:
                raise RuntimeError

            if args.reweight=='t':
                reweighting_t   = nn_input_t
            elif args.reweight=='t_norm':
                reweighting_t   = nn_input_t/(args.t0+args.t1)
            elif args.reweight=='sqr_t_norm':
                reweighting_t   = nn_input_t**2/((args.t0)**2+(args.t1)**2)
            elif args.reweight=='ones':
                reweighting_t   = torch.ones_like(nn_input_t)
            elif args.reweight=='sqr_t':
                reweighting_t   = nn_input_t**2
            else:
                raise RuntimeError
            x = x + eps

            if num_classes:
                y = y.cuda()
                mask = (torch.rand(y.size(0), device='cuda') < args.drop_label).int()
                y = (1 - mask) * y - mask
            else:
                y = None

            optimizer.zero_grad()
            loss, (z, outputs, logdets, mid_vars) = compute_loss(x, y,nn_input_t,reweighting_t,args.fp16)
            if dist.gather_concat(loss.view(1)).isnan().any():
                if dist.local_rank == 0:
                    print('nan detected, skipping step')
                continue
            if not args.nvp:
                model.update_prior(dist.gather_concat(z.detach().square().mean(dim=0, keepdim=True).sqrt()))
            dist.barrier()
            if args.loss_scaling:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            current_lr = lr_schedule.step()
            residuals,norms = mid_vars
            res_layer_mean = residuals.mean(dim=0)
            res_layer_norm_mean = norms.mean(dim=0)
            metrics.update({
                'loss': loss,
                'loss/mse(z)': 0.5 * (z**2).mean(),
                'loss/log(|det|)': logdets.mean(),
                'loss/residual_mean_sum': res_layer_mean.sum(),
                **{f'loss/residual_layer_{i}': v for i, v in enumerate(res_layer_mean)},
                **{f'loss/norm_layer_{i}': v for i, v in enumerate(res_layer_norm_mean)}
            })

            itr+=1
            if itr% 5 ==0 and dist.rank==0:
                writer.add_scalar(itr,'loss',loss.detach())
                writer.add_scalar(itr,'loss/mse(z)',0.5 * (z**2).mean().detach())
                writer.add_scalar(itr,'loss/log(|det|)',logdets.mean().detach())
                writer.add_scalar(itr,'lr',current_lr)

                for i, v in enumerate(res_layer_mean):
                    writer.add_scalar(itr, f'loss/residual_layer_{i}', v.detach())
                for i, v in enumerate(res_layer_norm_mean):
                    writer.add_scalar(itr, f'loss/norm_layer_{i}', v.detach())

                writer.add_scalar(itr, 'loss/residual_mean', residuals.sum().detach())
                writer.add_scalar(itr, 'loss/norm_mean', norms.sum().detach())

            if args.dry_run:
                break

        epoch_stat=metrics.compute(dist)
        metrics_dict = {'lr': current_lr, **epoch_stat}

        if dist.local_rank == 0:
            metrics.print(metrics_dict, epoch + 1)
            print('\tLayer norm', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))
            if dist.rank == 0:
                prefixed_epoch_stat = {f"epoch_{k}": v for k, v in epoch_stat.items()}
                writer.add_dict(itr,prefixed_epoch_stat)
            if (epoch + 1) % args.save_freq == 0:
                torch.save(model.state_dict(), args.logdir / f'{args.dataset}_model_{model_name}_{epoch + 1:03d}.pth')
                torch.save({'optimizer': optimizer.state_dict(), 'lr_schedule': lr_schedule.state_dict()},
                        args.logdir / f'{args.dataset}_opt_{model_name}_{epoch + 1:03d}.pth')
                if (epoch + 1) // args.save_freq > args.ckpts_to_keep:
                    epoch_to_remove = epoch + 1 - args.ckpts_to_keep * args.save_freq
                    os.remove(str(args.logdir / f'{args.dataset}_model_{model_name}_{epoch_to_remove:03d}.pth'))
                    os.remove(str(args.logdir / f'{args.dataset}_opt_{model_name}_{epoch_to_remove:03d}.pth'))
        dist.barrier()

        if (epoch + 1) % args.sample_freq == 0:
            model.eval()
            for i in range(args.num_samples // args.sample_batch_size):
                b = args.sample_batch_size // dist.world_size
                noise = fixed_noise[i * b : (i + 1) * b].to('cuda')
                y = None if fixed_y is None else fixed_y[i * b : (i + 1) * b].to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if args.fp16 else torch.float32):
                    with torch.no_grad():
                        _t=torch.ones(noise.shape[0],1,device=noise.device)*args.t1
                        samples = model.reverse(noise, y, t=_t, guidance=args.cfg,annealed_guidance=True, guide_flows=1)
                    samples = iterative_denoising(model, y, samples, args, dist, t0=args.t0, t1=args.t1, steps=10)

                assert isinstance(samples, torch.Tensor)
                with torch.no_grad():
                    if dist.world_size>8:
                        samples=0.5 * (samples.clip(min=-1, max=1) + 1)
                    else:
                        samples = 0.5 * (dist.gather_concat(samples.clip(min=-1, max=1)) + 1)
                    fid.update(samples, real=False)

                if args.dry_run:
                    break

            fid_score=fid.manual_compute(dist).item()

            if dist.local_rank == 0:
                utils.Metrics.print({'fid': fid_score}, epoch + 1)
                if dist.rank == 0:
                    writer.add_image(itr, 'generated img', tv.utils.make_grid(samples[0:64], nrow=8,scale_each=True))
                    writer.add_scalar(itr, 'fid', fid_score)

                # Save a grid for visual inspection
                tv.utils.save_image(samples, sample_dir / f'samples_{epoch+1:03d}.png', normalize=False, nrow=16)
            dist.barrier()

            torch.cuda.empty_cache()
            gc.collect()
            model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data', type=pathlib.Path)
    parser.add_argument('--logdir', default='runs', type=pathlib.Path)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'food101', 'imagenet', 'imagenet64', 'afhq', 'ffhq256','edm_imagenet','adm_imagenet'])
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--channel_size', default=3, type=int)

    parser.add_argument('--patch_sizes', default=[4], type=int, nargs='+')
    parser.add_argument('--blocks', default=4, type=int)
    parser.add_argument('--channels', default=[512], type=int, nargs='+')
    parser.add_argument('--layers_per_block', default=[8], type=int, nargs='+')
    parser.add_argument('--input_scale', default=1, type=float)
    parser.add_argument('--attn_dropout', default=0, type=float)
    parser.add_argument('--nvp', default=1, type=int)
    parser.add_argument('--cfg', default=0, type=float)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--drop_label', default=0, type=float)
    parser.add_argument('--sample_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--ckpts_to_keep', default=2, type=int)
    parser.add_argument('--num_samples', default=4096, type=int)
    parser.add_argument('--sample_batch_size', default=256, type=int)
    parser.add_argument('--denoising_batch_size', default=64, type=int)
    parser.add_argument('--ckpt_file', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--tag', default='none', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--fp16',default=1, type=int)
    parser.add_argument('--t0', default=1e-2, type=float)
    parser.add_argument('--t1', default=3e-1, type=float)
    parser.add_argument('--xflip',default=0, type=int)
    parser.add_argument('--reweight',default='ones', type=str)
    parser.add_argument('--t_sample',default='uni', type=str)

    parser.add_argument('--clip', default=1, type=int)
    parser.add_argument('--dyn', default='ode', type=str)
    parser.add_argument('--compile', default=0, type=int, help='Whether to use torch.compile')
    parser.add_argument('--loss_scaling', default=1, type=int, help='Whether to use AMP')
    parser.add_argument('--dry_run', default=0, type=int, help='Dry run for quick tests')
    parser.add_argument('--shared_backbone_layers', default=0, type=int, help='Number of shared backbone layers across MetaBlocks (0=no sharing)')
    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = f'{args.dataset}{args.img_size}_{args.channels}_{args.layers_per_block}_patch{args.patch_sizes}_ts{[args.t0,args.t1]}_reweight{args.reweight}_share{args.shared_backbone_layers}'

    main(args)
