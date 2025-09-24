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


class WindowMHSA1D(nn.Module):
    """1-D windowed MHSA over [B, N, D]; optional Swin-style shift; pads right then crops."""

    def __init__(self, dim, num_heads, window_size, shift=0,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.win = window_size
        self.shift = (shift % window_size) if window_size > 0 else 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # simple 1-D relative bias (helps local attention)
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, 2*window_size - 1))

    def forward(self, x):  # [B, N, D]
        B, N, D = x.shape
        if self.shift:
            x = torch.roll(x, shifts=self.shift, dims=1)
        pad = (self.win - (N % self.win)) % self.win
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Np, nW, w = x.shape[1], x.shape[1] // self.win, self.win
        xw = x.view(B, nW, w, D).reshape(B * nW, w, D)

        qkv = self.qkv(xw).reshape(B*nW, w, 3, self.num_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # add relative bias
        idx = torch.arange(w, device=x.device)
        rel = idx[None, :] - idx[:, None] + \
            (w - 1)               # [w,w] in [0..2w-2]
        attn = attn + self.rel_bias[:, rel].unsqueeze(0)          # [1,H,w,w]
        attn = self.attn_drop(attn.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B * nW, w, D)
        y = self.proj_drop(self.proj(y)).reshape(B, nW, w, D).reshape(B, Np, D)
        if pad:
            y = y[:, :N, :]
        if self.shift:
            y = torch.roll(y, shifts=-self.shift, dims=1)
        return y


class LocalBlock1D(nn.Module):
    """Pre-LN → WindowMHSA1D → +res → Pre-LN → MLP → +res (drop-in like Block)."""

    def __init__(self, dim, num_heads, window, shift=False, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)
        self.attn = WindowMHSA1D(dim, num_heads, window, shift=(window//2 if shift else 0),
                                 qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
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


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 64],
                 patch_size=[4, 64],
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
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings will be created lazily from actual (W, H) in forward
        self.pos_embed = None
        window_w = 12  # try 12 or 16
        self.blocks = nn.ModuleList([
            LocalBlock1D(self.embed_dim, num_heads, window=window_w, shift=False,
                         mlp_ratio=mlp_ratio, qkv_bias=True, drop=0.1, attn_drop=0.1,
                         norm_layer=norm_layer),
            LocalBlock1D(self.embed_dim, num_heads, window=window_w, shift=True,
                         mlp_ratio=mlp_ratio, qkv_bias=True, drop=0.1, attn_drop=0.1,
                         norm_layer=norm_layer),
            Block(self.embed_dim, num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, drop=0.1, attn_drop=0.1, norm_layer=norm_layer),
            Block(self.embed_dim, num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, drop=0.1, attn_drop=0.1, norm_layer=norm_layer),
        ])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # pos_embed is built dynamically from (W, H) in forward; do not pre-seed from grid_size

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

    def forward_features(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        x = self.patch_embed(x)                     # [B, C, W, H]
        b, c, W, H = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)       # [B, N, D]
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        self._ensure_pos_embed(W, H, x.device)
        if self.pos_embed is not None:
            x = x + self.pos_embed                  # [B, N, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False, return_features=False):
        feats = self.forward_features(
            x, mask_ratio, max_span_length, use_masking)  # [B, N, D]
        logits = self.head(feats)               # [B, N, nb_cls]  → CTC
        # keep your current post-norm if you like
        # logits = self.layer_norm(logits)
        if return_features:
            return logits, feats
        return logits

    def _ensure_pos_embed(self, W, H, device):
        """Build fixed sin-cos positional embedding from actual feature map size.
        Creates a non-trainable nn.Parameter of shape [1, W*H, D].
        """
        N = W * H
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
            pe = get_2d_sincos_pos_embed(self.embed_dim, [W, H])
            pe = torch.from_numpy(pe).float().unsqueeze(0).to(device)
            # store as a non-trainable parameter to keep prior behavior
            self.pos_embed = nn.Parameter(pe, requires_grad=False)


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
