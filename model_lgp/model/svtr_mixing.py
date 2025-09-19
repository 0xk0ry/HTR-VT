# ------------------ svtr_mixing.py ------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def _rearrange_bchw_to_bhwc(x):
    # B,C,H,W -> B,H,W,C
    return x.permute(0, 2, 3, 1).contiguous()

def _rearrange_bhwc_to_bchw(x):
    # B,H,W,C -> B,C,H,W
    return x.permute(0, 3, 1, 2).contiguous()

class GlobalMixing(nn.Module):
    """Full self-attention over all H*W tokens."""
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.drop_path = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_mult=4, drop=proj_drop)

    def forward(self, x_bchw):
        B, C, H, W = x_bchw.shape
        x = _rearrange_bchw_to_bhwc(x_bchw)          # B,H,W,C
        x = x.view(B, H*W, C)                         # B,N,C
        # block 1: MHA
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # B,N,C
        x = x + self.drop_path(attn_out)
        # block 2: MLP
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        # back to B,C,H,W
        x = x.view(B, H, W, C)
        return _rearrange_bhwc_to_bchw(x)

class LocalWindowAttention(nn.Module):
    """
    Sliding-window self-attention (non-overlapping windows for speed; you can
    enable overlap by padding/shift if you like). Window size defaults to (7,11) per SVTR.
    """
    def __init__(self, dim, num_heads=8, window_size: Tuple[int,int]=(7,11), attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        Wh, Ww = window_size
        assert Wh > 0 and Ww > 0
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_bhwc):
        # x: B,H,W,C (channel-last inside this function)
        B, H, W, C = x_bhwc.shape
        Wh, Ww = self.window_size

        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        x = F.pad(x_bhwc, (0,0, 0,pad_w, 0,pad_h))       # pad W then H in channel-last
        Hp, Wp = x.shape[1], x.shape[2]

        # partition into windows: (B, Hp/Wh, Wp/Ww, Wh, Ww, C) -> (B*nW, Wh*Ww, C)
        x = x.view(B, Hp//Wh, Wh, Wp//Ww, Ww, C).permute(0,1,3,2,4,5).contiguous()
        Bnw = B * (Hp//Wh) * (Wp//Ww)
        x = x.view(Bnw, Wh*Ww, C)

        # self-attention inside each window
        qkv = self.qkv(x)                                # (Bnw, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        head_dim = C // self.num_heads
        q = q.view(Bnw, -1, self.num_heads, head_dim).transpose(1,2)   # (Bnw,h,T,hd)
        k = k.view(Bnw, -1, self.num_heads, head_dim).transpose(1,2)
        v = v.view(Bnw, -1, self.num_heads, head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)           # (Bnw,h,T,T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                                 # (Bnw,h,T,hd)
        out = out.transpose(1,2).contiguous().view(Bnw, -1, C)         # (Bnw,T,C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # reverse windows back to (B,Hp,Wp,C)
        out = out.view(B, Hp//Wh, Wp//Ww, Wh, Ww, C).permute(0,1,3,2,4,5).contiguous()
        out = out.view(B, Hp, Wp, C)
        # remove padding
        if pad_h or pad_w:
            out = out[:, :H, :W, :].contiguous()
        return out

class LocalMixing(nn.Module):
    """Local windowed attention + MLP, with residuals."""
    def __init__(self, dim, num_heads=8, window_size=(7,11), drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lwa = LocalWindowAttention(dim, num_heads=num_heads, window_size=window_size, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_mult=4, drop=drop)

    def forward(self, x_bchw):
        # go channel-last for LN & linear ops
        x = _rearrange_bchw_to_bhwc(x_bchw)
        x = x + self.lwa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return _rearrange_bhwc_to_bchw(x)

class MergeH(nn.Module):
    """3x3 conv with stride (2,1) to halve height, keep width."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=(2,1), padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.bn(self.conv(x))

class SVTRLGMixer(nn.Module):
    """
    A compact SVTR-style mixer stage you can drop after your CNN features.
    By default: [Local x L] then [Global x G], optionally followed by merges.
    """
    def __init__(
        self,
        dim,                 # channels of incoming feature map
        num_heads_local=8,
        num_heads_global=8,
        window_size=(7,11),
        num_local=3,
        num_global=3,
        do_merge=False,
        dim_after_merge=None
    ):
        super().__init__()
        blocks = []
        for _ in range(num_local):
            blocks.append(LocalMixing(dim, num_heads=num_heads_local, window_size=window_size, drop=0.0))
        for _ in range(num_global):
            blocks.append(GlobalMixing(dim, num_heads=num_heads_global, proj_drop=0.0))
        self.blocks = nn.ModuleList(blocks)

        self.merge = None
        if do_merge:
            dim_out = dim_after_merge or dim
            self.merge = MergeH(dim, dim_out)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.merge is not None:
            x = self.merge(x)  # halves height
        return x
