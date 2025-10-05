import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial
import math
import random


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        self.bias = torch.ones(1, 1, self.num_patches, self.num_patches)
        self.back_bias = torch.triu(self.bias)
        self.forward_bias = torch.tril(self.bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            num_patches,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.0,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        self.attn = Attention(dim, num_patches, num_heads=num_heads,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(
            dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)


"""VAN components (Visual Attention Network) for collapsing height dimension.
Reference: Visual Attention Network (Li et al. 2022) – uses Large Kernel Attention (LKA).
We keep this lightweight since our input feature map is only H=4, W=128.
Goal here: (B, C, 4, 128)  ->  (B, C, 1, 128).
"""


class LargeKernelAttention(nn.Module):
    """Large Kernel Attention (simplified) used in VAN.
    Standard LKA uses depth-wise conv 5x5, then depth-wise dilated 7x7 (dilation=3), then pointwise 1x1.
    Here we retain that pattern; given very small height (4), receptive field easily covers it.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=5,
                            padding=2, groups=dim, bias=False)
        self.dwd = nn.Conv2d(dim, dim, kernel_size=7,
                             padding=9, dilation=3, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        attn = self.dw(x)
        attn = self.dwd(attn)
        attn = self.pw(attn)
        attn = self.bn(attn)
        return u * attn


class VANBlock(nn.Module):
    """A minimal VAN block: 1x1 -> GELU -> LKA -> 1x1 + residual."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.lka = LargeKernelAttention(dim)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj1(x)
        x = self.act(x)
        x = self.lka(x)
        x = self.proj2(x)
        x = self.norm(x)
        x = self.drop_path(x)
        return x + shortcut


class VANHeightReducer(nn.Module):
    """Stack a few VANBlocks then collapse height -> 1 via adaptive pooling.
    This module assumes input shape (B,C,H,W) with small H (e.g., 4).
    """

    def __init__(self, dim: int, depth: int = 2, drop_path: float = 0.0, pool: str = "avg"):
        super().__init__()
        self.blocks = nn.Sequential(
            *[VANBlock(dim, drop_path=drop_path) for _ in range(depth)])
        if pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(
                (1, None))  # keep width, collapse height
        elif pool == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, None))
        else:
            raise ValueError(f"Unsupported pool type: {pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        x = self.blocks(x)
        x = self.pool(x)  # (B,C,1,W)
        return x


class HorizontalMixer(nn.Module):
    """
    Depthwise 1xk conv along width + pointwise fuse, BN, GELU with residual.
    Operates on (B, C, 1, W) and preserves shape.
    """

    def __init__(self, dim: int, k: int = 9):
        super().__init__()
        pad = (k // 2)
        self.dw = nn.Conv2d(dim, dim, kernel_size=(
            1, k), padding=(0, pad), groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,1,W)
        shortcut = x
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        y = shortcut + y
        y = self.act(y)
        return y


class MaskedAutoencoderViT(nn.Module):

    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 64],
                 patch_size=[8, 32],
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 van_depth: int = 2,
                 van_pool: str = "avg",
                 van_drop_path: float = 0.0,
                 hmix_kernel: int = 9):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Initial positional embedding for the original (H*W) layout; we'll dynamically
        # recompute if sequence length changes after VAN height reduction.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        # VAN module to collapse height (e.g., 4 -> 1). We rely on downstream logic to
        # handle new sequence length (= width) after pooling.
        self.van_reducer = VANHeightReducer(
            embed_dim, depth=van_depth, drop_path=van_drop_path, pool=van_pool
        )
        self.hmix = HorizontalMixer(embed_dim, k=int(hmix_kernel))
        # Will lazily create a 1x1 projection if backbone channels != embed_dim
        self.proj_in = None
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # pos_embed = get_2d_sincos_pos_embed(self.embed_dim, [1, self.nb_query])
        # self.qry_tokens.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    # ---- MMS helpers ----
    # ---------------------------
    # 1-D Multiple Masking (MMS)
    # ---------------------------

    def _mask_random_1d(self, B: int, L: int, ratio: float, device) -> torch.Tensor:
        """Random token masking on 1-D sequence. Returns bool [B, L], True = masked."""
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        num = int(round(ratio * L))
        if num <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        noise = torch.rand(B, L, device=device)
        idx = noise.argsort(dim=1)[:, :num]         # per-sample masked indices
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return mask

    def _mask_block_1d(self, B: int, L: int, ratio: float, device,
                       min_block: int = 2) -> torch.Tensor:
        """
        Blockwise masking in 1-D (contiguous segments), no spacing constraints.
        Returns bool [B, L], True = masked.
        """
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        target = int(round(ratio * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            covered = int(mask[b].sum().item())
            # cap iterations to avoid infinite loops on tiny targets
            for _ in range(10000):
                if covered >= target:
                    break
                # choose a block length
                remain = max(1, target - covered)
                blk = random.randint(min_block, max(min_block, min(remain, L)))
                start = random.randint(0, max(0, L - blk))
                seg = mask[b, start:start+blk]
                prev = int(seg.sum().item())
                seg[:] = True
                covered += int(seg.sum().item()) - prev
        return mask

    def _mask_span_1d(self, B: int, L: int, ratio: float, max_span: int, device) -> torch.Tensor:
        if ratio <= 0.0 or max_span <= 0 or L <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        span_total = int(L * ratio)
        num_spans = span_total // max(1, max_span)
        if num_spans <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        s = min(max_span, L)  # fixed length (old behavior)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        for _ in range(num_spans):
            start = torch.randint(0, L - s + 1, (1,), device=device).item()
            mask[:, start:start + s] = True    # same start for the whole batch

        return mask

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:, idx:idx + max_span_length, :] = 0
        return mask

    # inside MaskedAutoencoderViT.forward_features(...)
    def forward_features(self, x, use_masking=False,
                         mask_mode="mms",   # "random" | "block" | "span_old" | "mms"
                         mask_ratio=0.5, max_span_length=8,
                         ratios=None, block_params=None):
        # [B,C,W,H] -> your [B,N,D] after reshape
        # --- Backbone (ResNet) ---
        x = self.patch_embed(x)              # (B, C, H=4, W=128) expected
        B, C, H, W = x.shape

        # If backbone channels differ from transformer embed_dim, project
        if C != self.embed_dim:
            if self.proj_in is None:
                # Create projection on the fly and move it to the same device as x
                self.proj_in = nn.Conv2d(
                    C, self.embed_dim, kernel_size=1, bias=False).to(x.device)
            x = self.proj_in(x)
            C = self.embed_dim

        # --- VAN Height Reduction (collapse H -> 1) ---
        x = self.van_reducer(x)             # (B, C, 1, W)
        # --- Horizontal local mixing along width (optional) ---
        x = self.hmix(x)                # (B, C, 1, W)
        x = x.squeeze(2)                    # (B, C, W)
        # (B, W, C) treat width as sequence length
        x = x.permute(0, 2, 1)
        N = x.size(1)

        if use_masking:
            if mask_mode == "random":
                keep = (~self._mask_random_1d(B, x.size(1),
                        mask_ratio, x.device)).float().unsqueeze(-1)
            elif mask_mode == "block":
                keep = (~self._mask_block_1d(B, x.size(1),
                        mask_ratio, x.device)).float().unsqueeze(-1)
            else:
                keep = (~self._mask_span_old_1d(B, x.size(1), mask_ratio,
                        max_span_length, x.device)).float().unsqueeze(-1)
            x = x * keep + (1 - keep) * self.mask_token  # [B,N,D]

        # pos + transformer as you already do
        # --- Positional Embedding ---
        if N == self.pos_embed.size(1):
            # Use precomputed (original) if sizes match (e.g., no height reduction case)
            pos = self.pos_embed[:, :N, :]
        else:
            # Dynamically create 2D sin-cos for (1, W) -> shape (W, C)
            dyn_pos = get_2d_sincos_pos_embed(
                self.embed_dim, [1, N])  # (1*W, C)
            pos = torch.from_numpy(dyn_pos).float().to(x.device).unsqueeze(0)
        x = x + pos

        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                           # [B,N,D]

    def forward(self, x, use_masking=False, return_features=False, mask_mode="mms", mask_ratio=None, max_span_length=None):
        feats = self.forward_features(
            x, use_masking=use_masking, mask_mode=mask_mode,
            mask_ratio=mask_ratio if mask_ratio is not None else 0.5,
            max_span_length=max_span_length if max_span_length is not None else 8)
        logits = self.head(feats)               # [B, N, nb_cls]  → CTC
        # keep your current post-norm if you like
        logits = self.layer_norm(logits)
        if return_features:
            return logits, feats
        return logits


def create_model(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 patch_size=(4, 64),
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 hmix_kernel=9,
                                 **kwargs)
    return model
