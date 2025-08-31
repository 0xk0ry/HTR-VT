import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath
import math

import numpy as np
from model import resnet18
from functools import partial


class Attention(nn.Module):
    """Multi-head self-attention with learnable relative position bias (default)."""
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
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * num_patches - 1), num_heads))
        coords = torch.arange(num_patches)
        relative_coords = coords[None, :] - coords[:, None]
        relative_coords += num_patches - 1
        self.register_buffer("relative_position_index", relative_coords, persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N]].permute(2, 0, 1)
        attn = attn + rel_bias.unsqueeze(0)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class ALiBiAttention(nn.Module):
    """Attention with ALiBi positional bias (additive linear bias by distance)."""
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.register_buffer('slopes', self._get_slopes(num_heads), persistent=False)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        def power_of_two_slopes(n):
            start = 2 ** (-8.0 / n)
            return torch.tensor([start ** i for i in range(n)], dtype=torch.float32)
        if math.log2(n_heads).is_integer():
            return power_of_two_slopes(n_heads)
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = power_of_two_slopes(closest)
        extra = power_of_two_slopes(2 * closest)[0::2][: n_heads - closest]
        return torch.cat([slopes, extra], dim=0)

    def _alibi_bias(self, N: int, device) -> torch.Tensor:
        pos = torch.arange(N, device=device)
        dist = (pos[None, :] - pos[:, None]).abs().float()  # (N,N)
        return (-self.slopes[:, None, None] * dist[None, None, :, :])  # (1,H,N,N)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._alibi_bias(N, x.device)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


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
            use_alibi: bool = False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)
        if use_alibi:
            self.attn = ALiBiAttention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
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


# ---------------- Conformer 1D Components ---------------- #
class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # (B, T, D)
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=9, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.pw_in = nn.Conv1d(dim, 2 * dim, 1)
        self.glu = nn.GLU(dim=1)
        pad = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.pw_out = nn.Conv1d(dim, dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):  # (B, T, D)
        y = self.ln(x).transpose(1, 2)  # (B, D, T)
        y = self.pw_in(y)
        y = self.glu(y)
        y = self.dw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw_out(y)
        y = self.do(y).transpose(1, 2)
        return y


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_patches, ffn_expansion=4, dropout=0.1, attn_drop=0.0, conv_kernel=9, qkv_bias=True, use_alibi: bool = False):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, expansion=ffn_expansion, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        if use_alibi:
            self.attn = ALiBiAttention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=dropout)
        else:
            self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=dropout)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, expansion=ffn_expansion, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)
        self.scale_ff = 0.5

    def forward(self, x):
        x = x + self.scale_ff * self.ff1(x)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(x)
        x = x + self.scale_ff * self.ff2(x)
        x = self.final_norm(x)
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

class HeadAdapterMLP(nn.Module):
    def __init__(self, d, r, p):
        super().__init__()
        self.ln = nn.LayerNorm(d, eps=1e-6)
        self.fc1 = nn.Linear(d, r)
        self.fc2 = nn.Linear(r, d)
        self.drop = nn.Dropout(p)
        self.act = nn.GELU()
    def forward(self, x):                 # x: (B, T, d)
        z = self.ln(x)
        y = self.fc2(self.drop(self.act(self.fc1(z))))
        return x + y                      # residual

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
                 norm_layer=nn.LayerNorm,
                 tone_classes: int = 6,
                 adapter_hidden_ratio: float = 0.5,
                 adapter_dropout: float = 0.1,
                 use_conformer: bool = False,
                 conformer_kernel: int = 9,
                 conformer_dropout: float = 0.1,
                 use_alibi: bool = False):
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
        # Remove absolute positional embedding (we rely on relative / ALiBi biases)
        if use_conformer:
            self.blocks = nn.ModuleList([
                ConformerBlock(embed_dim, num_heads, self.num_patches,
                                ffn_expansion=mlp_ratio, dropout=conformer_dropout,
                                attn_drop=0.0, conv_kernel=conformer_kernel, qkv_bias=True, use_alibi=use_alibi)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, self.num_patches,
                      mlp_ratio, qkv_bias=True, norm_layer=norm_layer, use_alibi=use_alibi)
                for _ in range(depth)
            ])
        self.use_conformer = use_conformer
        self.use_alibi = use_alibi

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        # Shared normalization before heads
        self.head = torch.nn.Linear(embed_dim, nb_cls)

        # Tone head adapter MLP
        self.tone_head = torch.nn.Linear(embed_dim, tone_classes)
        # adapter_hidden_dim = max(4, int(embed_dim * adapter_hidden_ratio))
        adapter_hidden_dim = max(4, self.embed_dim // 8)
        self.tone_adapter = HeadAdapterMLP(self.embed_dim, adapter_hidden_dim, adapter_dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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
        x = x.view(b, c, -1).permute(0, 2, 1)
        # masking: length -> length * mask_ratio
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        # No absolute pos_embed, rely on relative positional bias in Attention
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Base head: keep direct (most stable)
        base_logits = self.head(x)

        # Tone head through residual adapter
        tone_logits = self.tone_head(self.tone_adapter(x))

        return {"base": base_logits, "tone": tone_logits}


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
