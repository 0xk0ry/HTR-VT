# ---------------- lg_parallel.py ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def _tokens_to_bhwc(x_tokens, H, W):
    # x_tokens: [B, N, C] with N = H*W
    B, N, C = x_tokens.shape
    assert N == H * W, f"N={N} must equal H*W={H*W}"
    return x_tokens.view(B, H, W, C)

def _bhwc_to_tokens(x_bhwc):
    # x_bhwc: [B, H, W, C]
    B, H, W, C = x_bhwc.shape
    return x_bhwc.view(B, H*W, C)

class WindowMHSA(nn.Module):
    """Non-overlapping window self-attention on (H, W) with channel-last input."""
    def __init__(self, dim, num_heads=8, window_size: Tuple[int,int]=(7, 11), proj_drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_bhwc):
        B, H, W, C = x_bhwc.shape
        Wh, Ww = self.window_size
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        if pad_h or pad_w:
            x_bhwc = F.pad(x_bhwc, (0,0, 0,pad_w, 0,pad_h))  # pad W then H
        Hp, Wp = x_bhwc.shape[1], x_bhwc.shape[2]

        # (B, Hp/Wh, Wh, Wp/Ww, Ww, C) -> (B*nW, T, C)
        xw = x_bhwc.view(B, Hp//Wh, Wh, Wp//Ww, Ww, C).permute(0,1,3,2,4,5).contiguous()
        Bnw = B * (Hp//Wh) * (Wp//Ww)
        T = Wh * Ww
        xw = xw.view(Bnw, T, C)

        qkv = self.qkv(xw).chunk(3, dim=-1)  # (Bnw, T, C)
        q, k, v = qkv
        h = self.num_heads
        hd = C // h
        q = q.view(Bnw, T, h, hd).transpose(1, 2)
        k = k.view(Bnw, T, h, hd).transpose(1, 2)
        v = v.view(Bnw, T, h, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (Bnw, h, T, hd)
        out = out.transpose(1, 2).contiguous().view(Bnw, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # back to (B, Hp, Wp, C)
        out = out.view(B, Hp//Wh, Wp//Ww, Wh, Ww, C).permute(0,1,3,2,4,5).contiguous()
        out = out.view(B, Hp, Wp, C)
        if pad_h or pad_w:
            out = out[:, :H, :W, :].contiguous()
        return out

class GlobalMHSA(nn.Module):
    """Full self-attention over all pooled tokens with channel-last input, then project back."""
    def __init__(self, dim, num_heads=8, proj_drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x_bhwc):
        B, H, W, C = x_bhwc.shape
        x = x_bhwc.view(B, H*W, C)
        x = self.norm(x)
        y, _ = self.attn(x, x, x)  # (B, N, C)
        x = x + self.proj_drop(y)
        x = x + self.mlp(x)
        return x.view(B, H, W, C)

class LGParallelBlock(nn.Module):
    """
    Parallel Local-Global mixer:
      - Split channels: [C1 = r*C] -> Window-MHSA, [C2 = (1-r)*C] -> (pool -> Global-MHSA -> upsample)
      - Concat and project back to C. Residual add + MLP.
    """
    def __init__(
        self,
        embed_dim: int,
        grid_hw: Tuple[int,int],         # (H, W) of token grid
        r: float = 0.5,
        win_size: Tuple[int,int] = (7, 11),
        pool: Tuple[int,int] = (1, 2),   # (pool_h, pool_w) for the global branch
        heads_local: int = 8,
        heads_global: int = 8,
        drop: float = 0.0
    ):
        super().__init__()
        H, W = grid_hw
        self.H, self.W = H, W
        self.r = r
        c1 = int(round(embed_dim * r))
        c2 = embed_dim - c1

        self.pre_ln = nn.LayerNorm(embed_dim)

        # 1×1 splits
        self.split_local = nn.Linear(embed_dim, c1, bias=True)
        self.split_global = nn.Linear(embed_dim, c2, bias=True)

        # local path
        self.local_norm = nn.LayerNorm(c1)
        self.local_attn = WindowMHSA(dim=c1, num_heads=max(1, min(heads_local, max(1, c1 // 32))),
                                     window_size=win_size, proj_drop=drop, attn_drop=drop)

        # global path with pooling
        self.pool_hw = pool
        self.global_norm = nn.LayerNorm(c2)
        self.global_attn = GlobalMHSA(dim=c2, num_heads=max(1, min(heads_global, max(1, c2 // 32))),
                                      proj_drop=drop, attn_drop=drop)

        # fuse
        self.fuse = nn.Linear(c1 + c2, embed_dim)
        self.drop_path = nn.Dropout(drop)

        # post-MLP
        self.post_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(drop),
        )

    def _pool_hw(self, x_bhwc):
        if self.pool_hw == (1,1):
            return x_bhwc, self.H, self.W
        B, H, W, C = x_bhwc.shape
        ph, pw = self.pool_hw
        # average pool on spatial dims (H, W)
        x = x_bhwc.permute(0, 3, 1, 2)            # B,C,H,W
        x = F.avg_pool2d(x, kernel_size=(ph, pw), stride=(ph, pw), ceil_mode=True)
        H2, W2 = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1).contiguous()    # B,H2,W2,C
        return x, H2, W2

    def _upsample_to(self, x_bhwc, H, W):
        B, h, w, C = x_bhwc.shape
        if h == H and w == W:
            return x_bhwc
        x = x_bhwc.permute(0, 3, 1, 2)  # B,C,h,w
        x = F.interpolate(x, size=(H, W), mode="nearest")
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x_tokens):
        # x_tokens: [B, N, C], with N = H*W
        B, N, C = x_tokens.shape
        H, W = self.H, self.W

        # pre-norm & split channels
        x = self.pre_ln(x_tokens)
        x_l = self.split_local(x)     # [B, N, C1]
        x_g = self.split_global(x)    # [B, N, C2]

        # reshape to B,H,W,C*
        x_l = _tokens_to_bhwc(x_l, H, W)
        x_g = _tokens_to_bhwc(x_g, H, W)

        # Local: Window-MHSA
        xl = self.local_norm(_bhwc_to_tokens(x_l)).view(B, H, W, -1)  # ln in token space, then back
        xl = self.local_attn(xl)                                      # B,H,W,C1

        # Global: pool -> MHSA (full) -> upsample back
        xg_pooled, Hp, Wp = self._pool_hw(x_g)                        # B,Hp,Wp,C2
        xg = self.global_norm(_bhwc_to_tokens(xg_pooled))             # B, Hp*Wp, C2
        xg = self.global_attn(xg.view(B, Hp, Wp, -1))                 # B,Hp,Wp,C2
        xg = self._upsample_to(xg, H, W)                              # B,H,W,C2

        # Fuse and residual
        x_cat = torch.cat([xl, xg], dim=-1)                           # B,H,W,C1+C2
        x_out = self.fuse(x_cat)                                      # B,H,W,C
        x_out = _bhwc_to_tokens(x_out)                                # B,N,C
        x_out = x_tokens + self.drop_path(x_out)                      # residual 1

        # MLP + residual
        y = self.post_norm(x_out)
        y = self.mlp(y)
        return x_out + y
