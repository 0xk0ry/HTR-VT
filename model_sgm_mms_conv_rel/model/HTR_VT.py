import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial
import random
import re


class RelativePositionalAttention(nn.Module):
    """Multi-Head Attention with Relative Positional Embedding.
    
    This implements the relative positional embedding as described in:
    "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
    """
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., max_relative_position=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_patches = num_patches
        
        # Set max relative position based on sequence length if not provided
        if max_relative_position is None:
            # For 2D patches, we consider both H and W dimensions
            # Assuming square-ish patches, use sqrt as a reasonable default
            max_relative_position = int(np.sqrt(num_patches)) * 2
        self.max_relative_position = max_relative_position
        self.num_relative_distance = 2 * max_relative_position - 1
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Relative position embeddings for keys and values
        self.relative_position_k = nn.Parameter(
            torch.randn(self.num_relative_distance, self.head_dim)
        )
        self.relative_position_v = nn.Parameter(
            torch.randn(self.num_relative_distance, self.head_dim)
        )
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias table
        self._init_relative_position_bias_table()
        
    def _init_relative_position_bias_table(self):
        """Initialize the relative position bias table for 2D positions."""
        # Get relative position index for each patch pair
        coords_h = torch.arange(int(np.sqrt(self.num_patches)))
        coords_w = torch.arange(int(np.sqrt(self.num_patches)))
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        # Calculate relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        # Shift to start from 0
        relative_coords[:, :, 0] += int(np.sqrt(self.num_patches)) - 1
        relative_coords[:, :, 1] += int(np.sqrt(self.num_patches)) - 1
        relative_coords[:, :, 0] *= 2 * int(np.sqrt(self.num_patches)) - 1
        
        # Create relative position index
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def _get_relative_position_bias(self):
        """Get relative position bias from the bias table."""
        relative_position_bias = self.relative_position_k[self.relative_position_index.view(-1)].view(
            self.num_patches, self.num_patches, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is (B, num_heads, N, head_dim)
        
        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # Add relative position bias
        if hasattr(self, 'relative_position_index') and N == self.num_patches:
            relative_position_bias = self._get_relative_position_bias()
            attn = attn + relative_position_bias.unsqueeze(0)  # (B, num_heads, N, N)
        
        # Apply softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """Position-wise Feed Forward with configurable expansion (used for Conformer macaron)."""

    def __init__(self, dim, hidden_dim, dropout=0.1, activation=nn.SiLU):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lin2(self.act(self.lin1(x))))


class ConvModule(nn.Module):
    """Conformer convolution module (1D) implementing:
    LN -> pw conv -> GLU -> dw conv -> GroupNorm -> SiLU -> pw conv.
    Expect input (B, N, C).
    GroupNorm is used instead of BatchNorm for small batch stability.
    """

    def __init__(self, dim, kernel_size=3, dropout=0.1, drop_path=0.0, expansion=1.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        # smaller expansion to avoid huge internal bottleneck; expansion=1.0 means no expansion
        hidden_channels = int(dim * expansion)
        # pointwise conv: (B, C, N) -> (B, hidden, N)
        self.pointwise_conv1 = nn.Conv1d(dim, hidden_channels, kernel_size=1)
        self.glu = nn.GLU(dim=1) if hidden_channels % 2 == 0 else None
        # depthwise conv
        self.depthwise_conv = nn.Conv1d(
            hidden_channels // 2 if self.glu is not None else hidden_channels,
            hidden_channels // 2 if self.glu is not None else hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=(hidden_channels //
                    2 if self.glu is not None else hidden_channels),
            bias=True
        )
        # use GroupNorm for stable small-batch training
        self.norm = nn.GroupNorm(
            1, hidden_channels // 2 if self.glu is not None else hidden_channels, eps=1e-5)
        self.act = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            hidden_channels // 2 if self.glu is not None else hidden_channels, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: (B, N, C)
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.pointwise_conv1(x)  # (B, hidden, N)
        if self.glu is not None:
            x = self.glu(x)  # reduces to (B, hidden/2, N)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, N, C)
        return residual + self.drop_path(x)


class RelativePositionalConformerBlock(nn.Module):
    """Conformer encoder block with relative positional embedding.
    Order: x + 1/2 FFN -> x + Relative Pos MHSA -> x + ConvModule -> x + 1/2 FFN -> LayerNorm.
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_patches,
        mlp_ratio=4.0,
        ff_dropout=0.1,
        attn_dropout=0.0,
        conv_dropout=0.0,
        conv_kernel_size=3,
        norm_layer=nn.LayerNorm,
        drop_path=0.0,
        max_relative_position=None,
    ):
        super().__init__()
        ff_hidden = int(dim * mlp_ratio)
        
        # First macaron FFN
        self.ffn1_norm = norm_layer(dim, elementwise_affine=True)
        self.ffn1 = FeedForward(dim, ff_hidden, dropout=ff_dropout)
        
        # Relative positional attention
        self.attn_norm = norm_layer(dim, elementwise_affine=True)
        self.attn = RelativePositionalAttention(
            dim, 
            num_patches, 
            num_heads=num_heads,
            qkv_bias=True, 
            attn_drop=attn_dropout, 
            proj_drop=ff_dropout,
            max_relative_position=max_relative_position
        )
        
        # Convolution module
        self.conv_module = ConvModule(
            dim, kernel_size=conv_kernel_size, dropout=conv_dropout)
        
        # Second macaron FFN
        self.ffn2_norm = norm_layer(dim, elementwise_affine=True)
        self.ffn2 = FeedForward(dim, ff_hidden, dropout=ff_dropout)
        
        # Final normalization
        self.final_norm = norm_layer(dim, elementwise_affine=True)

        # Drop path (stochastic depth) on residual branches
        self.drop_path_ffn1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_attn = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_conv = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_ffn2 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # First macaron FFN (scaled by 1/2)
        x = x + 0.5 * self.drop_path_ffn1(self.ffn1(self.ffn1_norm(x)))
        
        # LayerNorm -> Multi-Head Attention with Relative Positional Embedding -> Dropout +
        x = x + self.drop_path_attn(self.attn(self.attn_norm(x)))
        
        # Conv module (already includes residual internally). Apply drop-path to the conv branch.
        conv_out = self.conv_module(x)
        conv_branch = conv_out - x
        x = x + self.drop_path_conv(conv_branch)
        
        # Second macaron FFN (scaled by 1/2)
        x = x + 0.5 * self.drop_path_ffn2(self.ffn2(self.ffn2_norm(x)))
        
        # Final norm (applied only once at the end)
        return self.final_norm(x)


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
    """HTR encoder with Conformer + Relative Positional Embedding."""

    def __init__(
        self,
        nb_cls=80,
        img_size=[512, 32],
        patch_size=[8, 32],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        max_relative_position: int = None,
    ):
        super().__init__()

        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # Conformer with relative positional embedding
        self.blocks = nn.ModuleList([
            RelativePositionalConformerBlock(
                embed_dim, num_heads, self.num_patches,
                mlp_ratio=mlp_ratio,
                ff_dropout=dropout, attn_dropout=dropout,
                conv_dropout=dropout, conv_kernel_size=conv_kernel_size,
                norm_layer=norm_layer, drop_path=dpr[i],
                max_relative_position=max_relative_position
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
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

    # --- Backward compatibility for loading older checkpoints ---
    def _upgrade_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Map legacy parameter names to the current module structure, in-place.
        - blocks.*.conv_layer_norm.* -> blocks.*.conv_module.layer_norm.*
        """
        if not isinstance(state_dict, dict):
            return state_dict

        remapped = {}
        to_delete = []

        # 0) Strip an optional leading 'module.' from DataParallel checkpoints
        has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
        if has_module_prefix:
            for k, v in list(state_dict.items()):
                new_k = k[len("module."):]
                if new_k not in state_dict:
                    remapped[new_k] = v
                to_delete.append(k)
            # apply moving keys
            for k in to_delete:
                state_dict.pop(k, None)
            state_dict.update(remapped)
            remapped.clear()
            to_delete.clear()

        # Direct string replace should be safe and faster than regex for this case
        conv_legacy_patterns = [
            (".conv_layer_norm.", ".conv_module.layer_norm."),
            (".conv_pointwise_conv1.", ".conv_module.pointwise_conv1."),
            (".conv_depthwise_conv.", ".conv_module.depthwise_conv."),
            (".conv_norm.", ".conv_module.norm."),
            (".conv_pointwise_conv2.", ".conv_module.pointwise_conv2."),
        ]

        for k, v in list(state_dict.items()):
            for old, new in conv_legacy_patterns:
                if old in k and new not in k:
                    new_k = k.replace(old, new)
                    # Only set if target key not already present
                    if new_k not in state_dict:
                        remapped[new_k] = v
                    to_delete.append(k)
                    break

        if remapped or to_delete:
            # apply removals then additions to avoid key overlap issues
            for k in to_delete:
                state_dict.pop(k, None)
            state_dict.update(remapped)

        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remap legacy keys before delegating to PyTorch loader
        upgraded = self._upgrade_state_dict_keys(dict(state_dict))
        return super().load_state_dict(upgraded, strict=strict)

    # ---- MMS helpers ----
    # ---------------------------
    # 1-D Multiple Masking (MMS)
    # ---------------------------

    def _mask_random_1d(self, B: int, L: int, ratio: float, device) -> torch.Tensor:
        """Random token masking on 1-D sequence. Returns bool [B, L], True = masked."""
        if ratio <= 0.0 or ratio > 1.0:
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
            # More reasonable upper bound
            max_iterations = min(10000, target * 3)
            for iteration in range(max_iterations):
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

                # Early exit if we're not making progress
                if iteration > 100 and covered < target * 0.1:
                    break
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
                # use k = span length (separates spans when ratio small)
                return None
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
                if s > L:
                    s = L
                l = random.randint(0, L - s)
                r = l + s - 1
                k = s if fixed_k is None else fixed_k
                # check spacing neighborhood
                left_ok = (l - k) < 0 or not used[max(0, l - k):l].any()
                right_ok = (
                    r + 1) >= L or not used[r+1:min(L, r + 1 + k)].any()
                if left_ok and right_ok:
                    used[l:r+1] = True
                    covered = int(used.sum().item())
            mask[b] = used
        return mask

    def _mask_span_old_1d(self, B: int, L: int, ratio: float, max_span: int, device) -> torch.Tensor:
        if ratio <= 0.0 or max_span <= 0 or L <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        span_total = int(L * ratio)
        num_spans = span_total // max(1, max_span)
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
            ratios = getattr(self, "mms_ratios", {
                             "random": 0.50, "block": 0.25, "span": 0.25})
        block_params = block_params or {}
        min_block = int(block_params.get("min_block", 2))

        m_rand = self._mask_random_1d(B, L, ratios.get(
            "random", 0.50), x.device)              # [B,L] True=masked
        m_block = self._mask_block_1d(B, L, ratios.get(
            "block", 0.25), x.device, min_block)     # [B,L]
        m_span = self._mask_span_1d(B, L, ratios.get(
            "span", 0.25), max_span_length, x.device)  # [B,L]

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
            mask[:, idx:idx + max_span_length, :] = 0
        return mask

    # inside MaskedAutoencoderViT.forward_features(...)
    def forward_features(self, x, use_masking=False,
                         mask_mode="mms",   # "random" | "block" | "span_old" | "mms"
                         mask_ratio=0.5, max_span_length=8,
                         ratios=None, block_params=None):
        # [B,C,W,H] -> your [B,N,D] after reshape
        x = self.patch_embed(x)
        B, C, W, H = x.shape
        # Ensure dimensions are correct before reshaping
        assert C == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {C}"
        x = x.view(B, C, -1).permute(0, 2, 1)         # [B,N,D]

        if use_masking:
            if mask_mode == "random":
                keep = (~self._mask_random_1d(B, x.size(1),
                        mask_ratio, x.device)).float().unsqueeze(-1)
            elif mask_mode == "block":
                keep = (~self._mask_block_1d(B, x.size(1),
                        mask_ratio, x.device)).float().unsqueeze(-1)
            elif mask_mode == "span_old":
                keep = (~self._mask_span_old_1d(B, x.size(1), mask_ratio,
                        max_span_length, x.device)).float().unsqueeze(-1)
            else:  # "mms" union (what you already have)
                keep = self.generate_mms_mask(
                    x, ratios=ratios, max_span_length=max_span_length, block_params=block_params)
            x = x * keep + (1 - keep) * self.mask_token  # [B,N,D]

        # pos + transformer as you already do
        x = x + self.pos_embed[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                           # [B,N,D]

    def forward(self, x, use_masking=False, return_features=False, mask_mode="mms", mask_ratio=None, max_span_length=None):
        feats = self.forward_features(
            # [B, N, D]
            x, use_masking=use_masking, mask_mode=mask_mode, mask_ratio=mask_ratio, max_span_length=max_span_length)
        logits = self.head(feats)               # [B, N, nb_cls]  → CTC
        # keep your current post-norm if you like
        logits = self.layer_norm(logits)
        if return_features:
            return logits, feats
        return logits


def create_model(nb_cls, img_size, mlp_ratio=4, **kwargs):
    model = MaskedAutoencoderViT(
        nb_cls,
        img_size=img_size,
        patch_size=(4, 64),
        embed_dim=768,
        depth=4,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
