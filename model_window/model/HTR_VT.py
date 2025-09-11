import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Relative positional bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * num_patches - 1), num_heads)
        )
        # Compute relative position index for each token pair
        coords = torch.arange(num_patches)
        relative_coords = coords[None, :] - coords[:, None]
        relative_coords += num_patches - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        if N > self.num_patches:
            raise ValueError(
                f"Sequence length N={N} exceeds configured num_patches={self.num_patches} for relative bias."
            )
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Add relative positional bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N]].permute(
            2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        # Apply key padding mask (True=valid, False=pad)
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)
            invalid_keys = ~key_padding_mask  # (B, N)
            attn = attn.masked_fill(invalid_keys.reshape(B, 1, 1, N), torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        if key_padding_mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
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
            norm_layer=nn.LayerNorm,
            window_size: int = 0,
            shift_size: int = 0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
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

    def _attend(self, x):
        if self.window_size <= 0:
            return self.attn(x)

        B, N, C = x.shape
        ws = self.window_size
        # zero-pad to multiple of ws
        pad = (ws - (N % ws)) % ws
        if pad:
            pad_zeros = torch.zeros(B, pad, C, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad_zeros], dim=1)
            N = x.size(1)

        # validity mask (True for real tokens)
        valid_mask = x.new_ones((B, N), dtype=torch.bool)
        if pad:
            valid_mask[:, -pad:] = False

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size,), dims=1)
            valid_mask = torch.roll(valid_mask, shifts=(-self.shift_size,), dims=1)

        # partition into windows and apply attention per window
        xw = _window_partition_1d(x, ws)  # [B, num_win, ws, C]
        mw = valid_mask.reshape(B, -1, ws)   # [B, num_win, ws]
        B_, num_win, ws_, C_ = xw.shape
        xw = xw.reshape(B_ * num_win, ws_, C_)
        mw = mw.contiguous().reshape(B_ * num_win, ws_)
        xw = self.attn(xw, key_padding_mask=mw)
        xw = xw.reshape(B_, num_win, ws_, C_)
        x = _window_reverse_1d(xw)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size,), dims=1)

        # remove pad
        if pad:
            x = x[:, : (N - pad), :]
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._attend(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# --- 1-D window helpers for windowed attention ---
def _window_partition_1d(x, ws):
    """Partition sequence x [B, N, C] into non-overlapping windows of size ws.
    Returns [B, num_win, ws, C]. Assumes N is a multiple of ws.
    """
    B, N, C = x.shape
    x = x.reshape(B, N // ws, ws, C)
    return x


def _window_reverse_1d(xw):
    """Reverse window partition. Input [B, num_win, ws, C] -> [B, N, C]."""
    B, num_win, ws, C = xw.shape
    return xw.contiguous().reshape(B, num_win * ws, C)


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


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 32],
                 patch_size=[8, 32],
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        # Set num_patches based on actual CNN output spatial dims (safe upper bound)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size[1], img_size[0])  # [B,C,H,W]
            feat = self.patch_embed(dummy)  # [1, C, H', W'] (or W', H')
            self.num_patches = int(feat.shape[2] * feat.shape[3])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Remove absolute positional embedding
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                self.num_patches,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                # Window first 1–2 blocks; rest global
                window_size=(16 if i in (0, 1) else 0),
                shift_size=(0 if i == 0 else (8 if i == 1 else 0)),
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:, idx:idx + max_span_length, :] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        # embed patches
        x = self.layer_norm(x)
        x = self.patch_embed(x)
        b, c, w, h = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        # masking: length -> length * mask_ratio
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        # No absolute pos_embed, rely on relative positional bias in Attention
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # To CTC Loss (do not LayerNorm logits)
        x = self.head(x)

        return x


def create_model(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 patch_size=(4, 64),
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model
