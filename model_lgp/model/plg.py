import math
from timm.models.vision_transformer import Mlp, DropPath
from timm.models.layers import DropPath, Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledGlobalMHSA(nn.Module):
    """
    Adaptive pool along width to a fixed token budget (g_tokens),
    run MHSA there, interpolate back, then scale by a single learnable α.
    Shapes in/out: [B, N, D] -> [B, N, D]
    """

    def __init__(self, dim, num_heads, g_tokens=64, pool='avg',
                 qkv_bias=True, attn_drop=0., proj_drop=0., alpha_init=0.4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.g_tokens = g_tokens
        assert pool in ('avg', 'max')
        self.pool = pool

        # attention projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mild channel norm inside the branch (stabilizes pooled stats)
        self.branch_norm = nn.LayerNorm(dim, elementwise_affine=False)

        # single learnable scalar α in (0,1) to keep global “quiet” initially
        # implemented as sigmoid(logit) so it stays in (0,1)
        logit = torch.log(torch.tensor(alpha_init) /
                          (1 - torch.tensor(alpha_init)))
        self.logit_alpha = nn.Parameter(logit)

    def forward(self, x):                  # x: [B, N, D]
        B, N, D = x.shape
        G = min(self.g_tokens, N)         # never exceed actual length

        # 1) pool width to G tokens: [B, D, N] -> [B, D, G]
        x_ch_first = x.transpose(1, 2)    # [B, D, N]
        if self.pool == 'avg':
            pooled = F.adaptive_avg_pool1d(x_ch_first, G)
        else:
            pooled = F.adaptive_max_pool1d(x_ch_first, G)

        # 2) MHSA on pooled tokens
        z = pooled.transpose(1, 2)        # [B, G, D]
        z = self.branch_norm(z)
        qkv = self.qkv(z).reshape(B, G, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, G, D)
        y = self.proj_drop(self.proj(y))  # [B, G, D]

        # 3) upsample back to N: [B, D, G] -> [B, D, N] -> [B, N, D]
        y = y.transpose(1, 2)             # [B, D, G]
        y = F.interpolate(y, size=N, mode="linear", align_corners=False)
        y = y.transpose(1, 2)             # [B, N, D]

        # 4) scale by α (learned scalar in (0,1))
        alpha = torch.sigmoid(self.logit_alpha)
        return y * alpha


class LayerScale(nn.Module):
    def __init__(self, dim, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        val = 1.0 if init_values is None else float(init_values)
        self.gamma = nn.Parameter(torch.ones(dim) * val)
        self.inplace = inplace

    def forward(self, x):
        return x * self.gamma


class WindowMHSA1D(nn.Module):
    """
    1-D windowed multi-head self-attention over a sequence [B, N, D].
    Pads on the RIGHT to a multiple of window size, then crops back.
    """

    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.win = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.shape
        w = self.win
        pad = (w - (N % w)) % w
        if pad:
            # pad tokens at the end (right)
            x = F.pad(x, (0, 0, 0, pad))

        Np = x.shape[1]
        nW = Np // w

        # [B, nW, w, D] -> [B*nW, w, D]
        xw = x.view(B, nW, w, D).reshape(B * nW, w, D)

        qkv = self.qkv(xw).reshape(B * nW, w, 3, self.num_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B * nW, w, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        # back to [B, Np, D]
        out = out.reshape(B, nW, w, D).reshape(B, Np, D)
        if pad:
            out = out[:, :N, :]
        return out


class GlobalMHSA(nn.Module):
    """
    Full (quadratic) MHSA over the entire sequence [B, N, D].
    Keep it simple for now; we can replace with pooled-global later.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class LocalGlobalParallelBlockSimple(nn.Module):
    def __init__(self, dim, num_heads, window_size=12, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, init_values=None,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 g_tokens=64, pool='avg', alpha_init=0.4):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        # local: your existing 1-D Window MHSA
        self.local_attn = WindowMHSA1D(dim, num_heads, window_size,
                                       qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # global: pooled MHSA with α-scaling (avg or max)
        self.global_attn = PooledGlobalMHSA(dim, num_heads, g_tokens=g_tokens, pool=pool,
                                            qkv_bias=qkv_bias, attn_drop=attn_drop,
                                            proj_drop=drop, alpha_init=alpha_init)

        self.fuse = nn.Linear(dim * 2, dim)

        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.dp1 = nn.Identity()  # keep DropPath off as you requested

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.dp2 = nn.Identity()

    def forward(self, x):  # [B, N, D]
        res = x
        y = self.norm1(x)
        y_loc = self.local_attn(y)           # [B, N, D]
        y_glb = self.global_attn(y)          # [B, N, D] (already scaled by α)
        y = torch.cat([y_loc, y_glb], dim=-1)
        y = self.fuse(y)                     # [B, N, D]
        x = res + self.ls1(y)

        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
