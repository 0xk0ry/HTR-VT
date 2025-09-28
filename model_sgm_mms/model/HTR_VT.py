import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial
import math, random


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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

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

        self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

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
                 img_size=[512, 32] ,
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
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
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
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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
    def _mask_random(self, B: int, L: int, ratio: float, device) -> torch.Tensor:
        """
        Random patch masking on 1-D sequence.
        Returns: bool mask [B, L] where True = masked
        """
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        num = int(round(ratio * L))
        if num <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        noise = torch.rand(B, L, device=device)
        idx = noise.argsort(dim=1)[:, :num]                   # per-sample indices to mask
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return mask

    def _mask_block(self, B: int, w: int, h: int, ratio: float,
                    device, min_block: int = 2, max_aspect: float = 3.0) -> torch.Tensor:
        """
        Blockwise masking on the 2-D token grid (BEiT-style rectangles).
        Returns: bool mask [B, L] where True = masked
        """
        if ratio <= 0.0:
            return torch.zeros(B, w*h, dtype=torch.bool, device=device)
        target = int(round(ratio * w * h))
        mask2d = torch.zeros(B, h, w, dtype=torch.bool, device=device)

        for b in range(B):
            covered = int(mask2d[b].sum().item())
            # cap iterations to avoid infinite loops on tiny targets
            for _ in range(10000):
                if covered >= target:
                    break
                # choose approximate block area
                remain = max(1, target - covered)
                area = random.randint(min_block, max(remain, min_block))
                # log-uniform aspect ratio in [1/max_aspect, max_aspect]
                ar = math.exp(random.uniform(-math.log(max_aspect), math.log(max_aspect)))
                bh = max(min_block, int(round(math.sqrt(area / ar))))
                bw = max(min_block, int(round(bh * ar)))
                bh, bw = min(bh, h), min(bw, w)
                y0 = random.randint(0, h - bh)
                x0 = random.randint(0, w - bw)
                block = mask2d[b, y0:y0+bh, x0:x0+bw]
                prev = int(block.sum().item())
                block[:] = True
                covered += int(block.sum().item()) - prev

        return mask2d.view(B, -1)

    def _mask_span(self, B: int, w: int, h: int, ratio: float,
                   max_span: int, device) -> torch.Tensor:
        """
        Span masking (Algorithm 1 in the paper), extended to multi-row grid:
        Mask *entire columns* over horizontal spans. Returns bool mask [B, L].
        """
        if ratio <= 0.0:
            return torch.zeros(B, w*h, dtype=torch.bool, device=device)

        # number of columns to cover (each column has h tokens)
        target_cols = int(round(ratio * w))
        mask2d = torch.zeros(B, h, w, dtype=torch.bool, device=device)

        # spacing k policy per Algorithm 1
        def spacing_for(R):
            if R <= 0.4:
                return None   # k = span length (set per-span)
            elif R <= 0.7:
                return 1
            else:
                return 0

        fixed_k = spacing_for(ratio)

        for b in range(B):
            used = torch.zeros(w, dtype=torch.bool, device=device)  # which columns are masked
            covered = int(used.sum().item())
            for _ in range(10000):
                if covered >= target_cols:
                    break
                s = random.randint(1, max(1, max_span))
                l = random.randint(0, max(0, w - s))
                r = l + s - 1
                k = s if fixed_k is None else fixed_k
                # ensure spacing neighborhood is clear
                left_ok  = (l - k) < 0 or not used[max(0, l - k):l].any()
                right_ok = (r + 1) >= w or not used[r+1:min(w, r + 1 + k)].any()
                if left_ok and right_ok:
                    used[l:r+1] = True
                    mask2d[b, :, l:r+1] = True
                    covered = int(used.sum().item())

        return mask2d.view(B, -1)

    def generate_mms_mask(self, x: torch.Tensor,
                          ratios: dict = None,
                          max_span_length: int = 8,
                          block_params: dict = None) -> torch.Tensor:
        """
        Build a *union* mask from three strategies: random, blockwise, span.
        x: [B, L, D] tokens.
        Returns: mask_keep float [B, L, 1], where 1=keep, 0=mask.
        """
        B, L, _ = x.shape
        # token grid from model config (must match your pos_embed grid)
        grid_h = self.grid_size[0]
        grid_w = self.grid_size[1]
        assert grid_h * grid_w == L, f"Grid {grid_h}x{grid_w} != sequence length {L}"

        # default ratios per paper (you can override by setting self.mms_ratios externally)
        if ratios is None:
            ratios = getattr(self, "mms_ratios", {"random": 0.75, "block": 0.50, "span": 0.50})
        block_params = block_params or {}
        min_block = block_params.get("min_block", 2)
        max_aspect = block_params.get("max_aspect", 3.0)

        m_rand  = self._mask_random(B, L, ratios.get("random", 0.75), x.device)                 # [B,L]
        m_block = self._mask_block(B, grid_w, grid_h, ratios.get("block", 0.50), x.device,
                                   min_block=min_block, max_aspect=max_aspect)                  # [B,L]
        m_span  = self._mask_span(B, grid_w, grid_h, ratios.get("span", 0.50), max_span_length, x.device)  # [B,L]

        m_union = (m_rand | m_block | m_span)        # True = masked anywhere
        mask_keep = (~m_union).float().unsqueeze(-1) # [B,L,1], 1=keep, 0=mask
        return mask_keep

    # ---- (kept for compatibility) single-span generator, now correctness-fixed ----
    def generate_span_mask(self, x, mask_ratio, max_span_length):
        """
        Backward-compatible span mask (now masks full columns across rows).
        Returns: mask_keep float [B, L, 1], where 1=keep, 0=mask.
        """
        B, L, _ = x.shape
        grid_h = self.grid_size[0]
        grid_w = self.grid_size[1]
        assert grid_h * grid_w == L, f"Grid {grid_h}x{grid_w} != sequence length {L}"
        m_span = self._mask_span(B, grid_w, grid_h, mask_ratio, max_span_length, x.device)  # [B,L], True=masked
        return (~m_span).float().unsqueeze(-1)

    def random_masking(self, x, mask_ratio, max_span_length):
        """
        MMS masking by default. To revert to single-span, set:
            self.use_mms = False
        """
        use_mms = getattr(self, "use_mms", True)
        if use_mms:
            mask = self.generate_mms_mask(x, ratios=getattr(self, "mms_ratios", None),
                                          max_span_length=max_span_length,
                                          block_params=getattr(self, "mms_block_params", None))  # [B,L,1]
        else:
            mask = self.generate_span_mask(x, mask_ratio, max_span_length)  # [B,L,1]
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
        x = x + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # To CTC Loss
        x = self.head(x)
        x = self.layer_norm(x)

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

