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
        """
        Span masking in 1-D (YOUR OLD SEMANTICS, but robust):
        - place contiguous spans of random length s ∈ [1, max_span]
        - enforce an Algorithm-1-like spacing policy via k depending on ratio
        - continue until ~ratio*L tokens are covered
        Returns bool [B, L], True = masked.
        """
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        L = int(L)
        max_span = int(max(1, min(max_span, L)))
        target = int(round(ratio * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        # spacing policy similar to Alg.1 (adapted to 1-D)
        def spacing_for(R):
            if R <= 0.4:
                return None   # use k = span length (separates spans when ratio small)
            elif R <= 0.7:
                return 1
            else:
                return 0
        fixed_k = spacing_for(ratio)

        for b in range(B):
            used = torch.zeros(L, dtype=torch.bool, device=device)
            covered = int(used.sum().item())
            for _ in range(10000):
                if covered >= target:
                    break
                s = random.randint(1, max_span)
                if s > L: s = L
                l = random.randint(0, L - s)
                r = l + s - 1
                k = s if fixed_k is None else fixed_k
                # check spacing neighborhood
                left_ok  = (l - k) < 0 or not used[max(0, l - k):l].any()
                right_ok = (r + 1) >= L or not used[r+1:min(L, r + 1 + k)].any()
                if left_ok and right_ok:
                    used[l:r+1] = True
                    covered = int(used.sum().item())
            mask[b] = used
        return mask

    def _mask_span_old_1d(self, B: int, L: int, ratio: float, max_span: int, device) -> torch.Tensor:
        if ratio <= 0.0 or max_span <= 0 or L <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        span_total = int(L * ratio)
        num_spans  = span_total // max(1, max_span)
        if num_spans <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        s = min(max_span, L)  # fixed length (old behavior)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        for _ in range(num_spans):
            start = torch.randint(0, L - s + 1, (1,), device=device).item()
            mask[:, start:start + s] = True    # same start for the whole batch

        return mask

    def generate_mms_mask(self, x: torch.Tensor,
                        ratios: dict = None,
                        max_span_length: int = 8,
                        block_params: dict = None) -> torch.Tensor:
        """
        Build UNION of three 1-D masks: random, blockwise, span.
        x: [B, L, D] tokens.
        Returns: mask_keep float [B, L, 1], where 1=keep, 0=mask.
        """
        B, L, _ = x.shape
        if ratios is None:
            ratios = getattr(self, "mms_ratios", {"random": 0.50, "block": 0.25, "span": 0.25})
        block_params = block_params or {}
        min_block = int(block_params.get("min_block", 2))

        m_rand  = self._mask_random_1d(B, L, ratios.get("random", 0.50), x.device)              # [B,L] True=masked
        m_block = self._mask_block_1d(B, L, ratios.get("block", 0.25), x.device, min_block)     # [B,L]
        m_span  = self._mask_span_1d(B, L, ratios.get("span", 0.25), max_span_length, x.device) # [B,L]

        m_union = (m_rand | m_block | m_span)   # True = masked by any strategy
        mask_keep = (~m_union).float().unsqueeze(-1)  # [B,L,1], 1=keep, 0=mask
        return mask_keep


    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:,idx:idx + max_span_length,:] = 0
        return mask

    # inside MaskedAutoencoderViT.forward_features(...)
    def forward_features(self, x, use_masking=False,
                        mask_mode="mms",   # "random" | "block" | "span_old" | "mms"
                        mask_ratio=0.5, max_span_length=8,
                        ratios=None, block_params=None):
        x = self.patch_embed(x)                       # [B,C,W,H] -> your [B,N,D] after reshape
        B, C, W, H = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)         # [B,N,D]

        if use_masking:
            if mask_mode == "random":
                keep = (~self._mask_random_1d(B, x.size(1), mask_ratio, x.device)).float().unsqueeze(-1)
            elif mask_mode == "block":
                keep = (~self._mask_block_1d(B, x.size(1), mask_ratio, x.device)).float().unsqueeze(-1)
            elif mask_mode == "span_old":
                keep = (~self._mask_span_old_1d(B, x.size(1), mask_ratio, max_span_length, x.device)).float().unsqueeze(-1)
            else:  # "mms" union (what you already have)
                keep = self.generate_mms_mask(x, ratios=ratios, max_span_length=max_span_length, block_params=block_params)
            x = x * keep + (1 - keep) * self.mask_token  # [B,N,D]

        # pos + transformer as you already do
        x = x + self.pos_embed[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                           # [B,N,D]


    def forward(self, x, use_masking=False, return_features=False, mask_mode="mms", mask_ratio=None, max_span_length=None):
        feats = self.forward_features(
            x, use_masking=use_masking, mask_mode=mask_mode, mask_ratio=mask_ratio, max_span_length=max_span_length)  # [B, N, D]
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
                                 **kwargs)
    return model
