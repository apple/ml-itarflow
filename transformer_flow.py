#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import torch
import math
class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SDPA: bool = True

    def __init__(self, in_channels: int, head_channels: int, attn_dropout: float = 0):
        assert in_channels % head_channels == 0
        super().__init__()
        self.attn_dropout = attn_dropout
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward_sdpa(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, dropout_p=self.attn_dropout if self.training else 0)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SDPA:
            return self.forward_sdpa(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion), torch.nn.GELU(), torch.nn.Linear(channels * expansion, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4, attn_dropout: float = 0):
        super().__init__()
        self.attention = Attention(channels, head_channels, attn_dropout)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


# ---------------- Fourier time embedding ---------------- 
class FourierTimeEmbedding(torch.nn.Module):  
    def __init__(
        self,
        channels: int,
        max_period: float = 10000.0,
    ):  
        super().__init__()
        self.channels = channels
        self.max_period = max_period
        self.mlp = torch.nn.Sequential(  
            torch.nn.Linear(channels, channels * 4),  
            torch.nn.GELU(),  
            torch.nn.Linear(channels * 4, channels),  
        )  

    def forward(self, t: torch.Tensor) -> torch.Tensor:  
        assert t.dim()==2
        t = t.view(-1)
        half = self.channels // 2
        if half == 0:
            emb = t[:, None]
        else:
            freqs = torch.exp(-math.log(self.max_period) * torch.arange(
                half, device=t.device, dtype=t.dtype) / max(half, 1)
            )
            args = t[:, None] * freqs[None, :]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if emb.size(-1) < self.channels:
                emb = torch.nn.functional.pad(emb, (0, self.channels - emb.size(-1)))
        return self.mlp(emb)


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
        attn_dropout: float = 0,
        shared_backbone: torch.nn.ModuleList | None = None,
        num_head_layers: int = 0,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if num_classes:
            self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None

        # Shared backbone (not owned by this module) + per-block head layers
        self.shared_backbone = shared_backbone
        if shared_backbone is not None:
            self.head_blocks = torch.nn.ModuleList(
                [AttentionBlock(channels, head_dim, expansion, attn_dropout) for _ in range(num_head_layers)]
            )
        else:
            # Legacy path: all layers owned by this block
            self.head_blocks = torch.nn.ModuleList(
                [AttentionBlock(channels, head_dim, expansion, attn_dropout) for _ in range(num_layers)]
            )

        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))
        self.time_embed = FourierTimeEmbedding(channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, t: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        bs,l,c = x.shape
        x = self.proj_in(x) + pos_embed
        if t is not None:  
            t_embed = self.time_embed(t)[:, None, :].type_as(x)  
            x = x + t_embed  
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        if self.shared_backbone is not None:
            for block in self.shared_backbone:
                x = block(x, self.attn_mask)
        for block in self.head_blocks:
            x = block(x, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        logdet= -xa.mean(dim=[1, 2])
        return self.permutation((x_in - xb) * scale, inverse=True), logdet


    def reverse_step(
        self, x: torch.Tensor, pos_embed: torch.Tensor, i: int, y: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond', t: torch.Tensor | None = None  
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]  # get i-th patch but keep the sequence dimension
        x = self.proj_in(x_in) + pos_embed[i : i + 1]
        if t is not None:  
            t_embed = self.time_embed(t)[:, None, :].type_as(x)  
            x = x + t_embed  
        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        if self.shared_backbone is not None:
            for block in self.shared_backbone:
                x = block(x, attn_temp=attn_temp, which_cache=which_cache)
        for block in self.head_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        modules_to_scan = list(self.modules())
        if self.shared_backbone is not None:
            modules_to_scan += list(self.shared_backbone.modules())
        for m in modules_to_scan:
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        t: torch.Tensor | None = None,  
        lerp_style: str = 'tarflow',
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1)
        bs=x.shape[0]
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond', t=t)  
            za=za[:, 0]
            scale = (za.float()).exp().type(za.dtype)

            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond', t=t)  
                za_u = za_u[:,0]
                scale_u = (za_u.float()).exp().type(za_u.dtype)

                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                zb,scale = self.guidance(scale,zb,scale_u,zb_u,guidance=g,lerp_type=lerp_style)

            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


    def guidance(self, za, zb, za_u,zb_u,guidance, r=1.0, guide_what='ab',lerp_type='starflow'):
        g = r * guidance
        
        def logits_guided(mu_c, sigma_c, mu_u, sigma_u, w):
            # inspired from: (1+w) * logP_cond - w * logP_uncond
            # sigma_c = torch.minimum(sigma_c, sigma_u)
            s = (sigma_c / sigma_u).clip(max=1.0).square()
            sigma_eff = sigma_c / (1 + w - w * s).sqrt()
            mu_eff = ((1 + w) * mu_c - (w * s) * mu_u) / (1 + w - w * s)   
            return mu_eff, sigma_eff
        
        def original_guidance(mu_c, sigma_c, mu_u, sigma_u, w):
            if 'a' in guide_what:
                sigma_c = sigma_c + g * (sigma_c - sigma_u)
            if 'b' in guide_what:
                mu_c = mu_c + g * (mu_c - mu_u)
            return mu_c, sigma_c

        #zb, za = original_guidance(zb, za, zb_u, za_u, guidance)
        lerp =  {
                    'starflow':logits_guided,
                    'tarflow':original_guidance
                }.get(lerp_type)
        zb, za = lerp(zb, za, zb_u, za_u, guidance)
        return zb, za
    
class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_sizes: list,
        channels: list,
        num_blocks: int,
        layers_per_block: list,
        nvp: bool = True,
        num_classes: int = 0,
        input_scale: float = 1,
        attn_dropout: float = 0,
        shared_backbone_layers: int = 0,
    ):
        super().__init__()
        self.input_scale = input_scale
        self.img_size = img_size
        if len(patch_sizes) == 1:
            patch_sizes = patch_sizes * num_blocks
        assert len(patch_sizes) == num_blocks
        self.patch_sizes = patch_sizes
        if len(layers_per_block) == 1:
            layers_per_block = layers_per_block * num_blocks
        assert len(layers_per_block) == num_blocks
        if len(channels) == 1:
            channels = channels * num_blocks
        assert len(channels) == num_blocks

        # Create shared backbone if requested
        if shared_backbone_layers > 0:
            self.shared_backbone = torch.nn.ModuleList([
                AttentionBlock(channels[0], 64, 4, attn_dropout)
                for _ in range(shared_backbone_layers)
            ])
        else:
            self.shared_backbone = None

        blocks = []
        for i in range(num_blocks):
            num_patches = (img_size // patch_sizes[i]) ** 2
            head_layers = max(0, layers_per_block[i] - shared_backbone_layers) if self.shared_backbone is not None else 0
            blocks.append(
                MetaBlock(
                    in_channels * patch_sizes[i]**2,
                    channels[i],
                    num_patches,
                    PermutationIdentity(num_patches) if (i % 2) == 0 else PermutationFlip(num_patches),
                    layers_per_block[i],
                    nvp=nvp,
                    num_classes=num_classes,
                    attn_dropout=attn_dropout,
                    shared_backbone=self.shared_backbone,
                    num_head_layers=head_layers,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        self.register_buffer('var', torch.ones(num_patches, in_channels * patch_sizes[-1]**2))

    def patchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, patch_size, stride=patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), patch_size, stride=patch_size)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None, t: torch.Tensor | None = None  
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x = self.patchify(x, self.patch_sizes[0]) * self.input_scale
        t=torch.log(t)
        outputs     = []
        logdets     = torch.zeros((), device=x.device) + math.log(self.input_scale)
        residuals   = []
        norms       = []
        # residuals   = torch.zeros((), device=x.device) + np.log(self.input_scale)
        for i, block in enumerate(self.blocks):
            if i > 0 and self.patch_sizes[i] != self.patch_sizes[i - 1]:
                x = self.patchify(self.unpatchify(x, self.patch_sizes[i - 1]), self.patch_sizes[i])
            x_out, logdet = block(x, y, t=t)  

            residual    = ((x_out-x)**2).mean(dim=[1,2])
            norm        = (x_out**2).mean(dim=[1,2])
            residuals.append(residual)
            norms.append(norm)
            x=x_out
            logdets = logdets + logdet
            outputs.append(x)

        residuals   = torch.stack(residuals, dim=1)
        norms       = torch.stack(norms, dim=1)
        return x, outputs, logdets, [residuals,norms]

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor, mid_vars: torch.Tensor, t: torch.Tensor):
        t=t.reshape(z.shape[0])
        loss = 0.5 * z.pow(2).mean(dim=[1, 2]) - logdets
        loss=(loss*t).mean()
        return loss

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
        guide_flows: int = -1,
        t: torch.Tensor | None = None,  
        lerp_style: str = 'tarflow',
    ) -> torch.Tensor | list[torch.Tensor]:
        seq = [self.unpatchify(x, self.patch_sizes[-1])]
        x = x * self.var.sqrt()
        t=torch.log(t)

        num_blocks = len(self.blocks)
        if guide_flows < 0:
            guide_flows = num_blocks
        if lerp_style == 'starflow': 
            guide_flows = 1
            # annealed_guidance = 0

        for i, block in enumerate(reversed(self.blocks)):
            if lerp_style == 'starflow' and guide_flows == -1 and i>0:
                lerp_style = 'tarflow'
            if i > 0 and self.patch_sizes[num_blocks - 1 - i] != self.patch_sizes[num_blocks - i]:
                x = self.patchify(self.unpatchify(x, self.patch_sizes[num_blocks - i]), self.patch_sizes[num_blocks - 1 - i])

            if i < guide_flows:
                g = guidance
            else:
                g = 0
            x = block.reverse(x, y, g, guide_what, attn_temp, annealed_guidance, t=t,lerp_style=lerp_style)  
            seq.append(self.unpatchify(x, self.patch_sizes[num_blocks - 1 - i]))
        x = self.unpatchify(x, self.patch_sizes[0]) / self.input_scale
        if not return_sequence:
            return x
        else:
            return seq

    def reset_attn_mask(self): 
        for m in self.modules():
            if isinstance(m, MetaBlock):
                m.attn_mask.data.copy_(torch.tril(torch.ones_like(m.attn_mask)))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('.attn_blocks.', '.head_blocks.')
            new_state_dict[new_key] = v
        super().load_state_dict(new_state_dict, strict, assign)
        self.reset_attn_mask()