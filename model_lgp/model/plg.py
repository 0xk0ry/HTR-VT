import math
from timm.models.vision_transformer import Mlp, DropPath
from timm.models.layers import DropPath, Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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
    """
    Pre-LN → [Local(Window MHSA) || Global(MHSA)] → Concat → Linear(2D→D) → +res
    Pre-LN → MLP → +res
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=12,          # 1-D local window size
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,        # optional LayerScale
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # Norm before attention
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        # Parallel attentions
        self.local_attn = WindowMHSA1D(
            dim, num_heads, window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.global_attn = GlobalMHSA(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # Fuse (concat channels then project back to D)
        self.fuse = nn.Linear(dim * 2, dim)

        # Residual helpers
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.dp1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # FFN path
        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(
            dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.dp2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):  # x: [B, N, D]
        # --- Parallel attention + fusion ---
        res = x
        y = self.norm1(x)
        y_local = self.local_attn(y)      # [B, N, D]
        y_global = self.global_attn(y)     # [B, N, D]
        y = torch.cat([y_local, y_global], dim=-1)  # [B, N, 2D]
        y = self.fuse(y)                                   # [B, N, D]
        x = res + self.dp1(self.ls1(y))

        # --- MLP ---
        x = x + self.dp2(self.ls2(self.mlp(self.norm2(x))))
        return x
