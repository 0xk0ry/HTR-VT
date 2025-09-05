import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model_conformer.model import resnet18
from functools import partial


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        # Some callers (Conformer MHSA) pass num_patches=None; skip bias maps in that case.
        if self.num_patches is not None:
            self.bias = torch.ones(1, 1, self.num_patches, self.num_patches)
            self.back_bias = torch.triu(self.bias)
            self.forward_bias = torch.tril(self.bias)
        else:
            self.bias = None
            self.back_bias = None
            self.forward_bias = None
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


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, nb_cls=80, img_size=[32, 512], patch_size=[4, 32],
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 encoder_type="vit",
                 conformer_kernel=15,
                 conformer_ratio=1.0,
                 dropout=0.1, attn_drop=0.0, drop_path=0.0,
                 apply_logit_layer_norm=False):
        super().__init__()
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.apply_logit_layer_norm = apply_logit_layer_norm

        self.blocks = nn.ModuleList()
        num_conformer = int(depth * conformer_ratio) if encoder_type == "hybrid" else (depth if encoder_type == "conformer" else 0)
        for i in range(depth):
            if (encoder_type == "conformer") or (encoder_type == "hybrid" and i < num_conformer):
                self.blocks.append(
                    ConformerBlock(embed_dim, num_heads, ffn_mult=mlp_ratio, conv_kernel=conformer_kernel,
                                   dropout=dropout, attn_drop=attn_drop, drop_path=drop_path)
                )
            else:
                self.blocks.append(
                    Block(embed_dim, num_heads, self.num_patches, mlp_ratio, qkv_bias=True,
                          drop=dropout, attn_drop=attn_drop, norm_layer=norm_layer, drop_path=drop_path)
                )
        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)

        self._init_position_and_weights()

    def _init_position_and_weights(self):
        self._set_pos_embed(self.grid_size)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _set_pos_embed(self, grid_size):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size)
        self.pos_embed.data.resize_(1, grid_size[0] * grid_size[1], self.embed_dim)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.grid_size = list(grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def _maybe_recompute_pos_embed(self, h_tokens, w_tokens, debug=False):
        if h_tokens * w_tokens != self.num_patches:
            if debug:
                print(f"[SANITY] Recomputing pos_embed: old grid {self.grid_size} -> new grid {[h_tokens, w_tokens]}")
            self._set_pos_embed([h_tokens, w_tokens])

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape
        mask = torch.ones(N, L, 1, device=x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for _ in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,), device=x.device)
            mask[:, idx:idx + max_span_length, :] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        return x * mask + (1 - mask) * self.mask_token

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False, debug=False):
        x = self.patch_embed(x)
        b, c, h_cnn, w_cnn = x.shape
        if debug:
            print(f"[SANITY] cnn_out: {x.shape} -> tokens grid {h_cnn}x{w_cnn}={h_cnn*w_cnn}")
        seq_len = h_cnn * w_cnn
        self._maybe_recompute_pos_embed(h_cnn, w_cnn, debug)
        x = x.view(b, c, -1).permute(0, 2, 1)
        if debug:
            print(f"[SANITY] seq: {x.shape}")
        assert x.size(1) == seq_len
        if use_masking and mask_ratio > 0:
            x = self.random_masking(x, mask_ratio, max_span_length)
        if self.pos_embed.shape[1] != x.shape[1]:
            raise ValueError("pos_embed length mismatch after recompute")
        x = x + self.pos_embed
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if debug and i == 0:
                print(f"[SANITY] after first block: {x.shape}")
        x = self.norm(x)
        x = self.head(x)
        if self.apply_logit_layer_norm:
            if debug:
                print(f"[SANITY] logits pre-LN: {x.shape}")
            x = self.layer_norm(x)
            if debug:
                print(f"[SANITY] logits post-LN: {x.shape}")
        elif debug:
            print(f"[SANITY] logits: {x.shape} (no post-head LayerNorm)")
        return x


class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class ConvModule(nn.Module):
    """
    1-D conformer conv block:
      LayerNorm -> PW (GLU, expand 2x) -> DWConv1d(k) -> BN -> Swish -> PW -> Dropout -> +residual
    Expects [B, T, D]. DWConv is along T.
    """

    def __init__(self, dim, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, dim)   # we do GLU with a Linear (2x inside)
        self.glu = GLU(dim)
        self.dw = nn.Conv1d(dim, dim, kernel_size,
                            padding=kernel_size//2, groups=dim)
        self.bn = nn.BatchNorm1d(dim, eps=1e-5)
        self.act = Swish()
        self.pw2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        residual = x
        x = self.ln(x)
        x = self.glu(x)                  # [B,T,D]
        x = x.transpose(1, 2)            # [B,D,T]
        x = self.dw(x)                   # depthwise conv over T
        x = self.bn(x)
        x = x.transpose(1, 2)            # [B,T,D]
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x + residual


class FeedForward(nn.Module):
    """ Macaron FFN piece. """

    def __init__(self, dim, mult=4.0, dropout=0.1, act=nn.GELU):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, hidden),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)


class MHSA(nn.Module):
    """ reuse your Attention but pre-norm & dropout-ready """

    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_patches=None, num_heads=num_heads, qkv_bias=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        return x + self.attn(self.ln(x))


class ConformerBlock(nn.Module):
    """
    Macaron-style Conformer:
      x = x + 0.5*FFN1
      x = x + MHSA
      x = x + ConvModule
      x = x + 0.5*FFN2
    """

    def __init__(self, dim, num_heads, ffn_mult=4.0, conv_kernel=15,
                 dropout=0.1, attn_drop=0., drop_path=0.0):
        super().__init__()
        self.ff1 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.mhsa = MHSA(dim, num_heads=num_heads,
                         attn_drop=attn_drop, proj_drop=dropout)
        self.conv = ConvModule(dim, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.dp3 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.dp4 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + 0.5 * self.dp1(self.ff1(x))
        x = self.dp2(self.mhsa(x))       # mhsa already residuals inside
        x = self.dp3(self.conv(x))       # conv residual inside
        x = x + 0.5 * self.dp4(self.ff2(x))
        return x


def create_model(nb_cls, img_size, **kwargs):
    # Example call:
    # all conformer
    # model = create_model(nb_cls, img_size, encoder_type="conformer", depth=8, num_heads=6)

    # hybrid: first half conformer, then ViT
    # model = create_model(nb_cls, img_size, encoder_type="hybrid", conformer_ratio=0.5)

    return MaskedAutoencoderViT(
        nb_cls,
        img_size=img_size,
        # Keep patch_size aligned so that grid_size product matches CNN output tokens (expected 8x8=64 tokens).
        # Original (4,64) led to mismatch with width < patch_size_w. Using (8,32) for 512x32 input -> 64 tokens.
        patch_size=(4, 32),
        embed_dim=768,
        depth=4,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # 'vit' | 'conformer' | 'hybrid'
        encoder_type=kwargs.get("encoder_type", "vit"),
        conformer_kernel=kwargs.get("conformer_kernel", 15),
        conformer_ratio=kwargs.get("conformer_ratio", 1.0),
        dropout=kwargs.get("dropout", 0.1),
        attn_drop=kwargs.get("attn_drop", 0.0),
        drop_path=kwargs.get("drop_path", 0.0),
    )

    return model
