# ----- SWIN UTILS -----
from model import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# window partition/reverse for (H, W)


def window_partition(x, wh, ww):
    # x: [B, H, W, C]
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh*ww, C)
    return windows  # [B * (H/wh) * (W/ww), wh*ww, C]


def window_reverse(windows, wh, ww, H, W, B):
    # windows: [B * num_wins, wh*ww, C]
    nw_h = H // wh
    nw_w = W // ww
    x = windows.view(B, nw_h, nw_w, wh, ww, -1).permute(0,
                                                        1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# ----- RELATIVE POSITION BIAS -----


class WindowAttention2D(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.wh, self.ww = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*self.wh-1)*(2*self.ww-1), num_heads)
        )
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.wh)
        coords_w = torch.arange(self.ww)
        coords = torch.stack(torch.meshgrid(
            coords_h, coords_w, indexing='ij'))  # [2, wh, ww]
        coords_flat = torch.flatten(coords, 1)  # [2, wh*ww]
        relative_coords = coords_flat[:, :, None] - \
            coords_flat[:, None, :]  # [2, wh*ww, wh*ww]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()       # [wh*ww, wh*ww, 2]
        relative_coords[:, :, 0] += self.wh - 1
        relative_coords[:, :, 1] += self.ww - 1
        relative_coords[:, :, 0] *= 2*self.ww - 1
        # [wh*ww, wh*ww]
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, attn_mask=None):
        # x: [num_windows*B, wh*ww, C]
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(
            Bn, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [Bn, N, h, d]

        q = q.permute(0, 2, 1, 3)  # [Bn, h, N, d]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [Bn, h, N, N]

        # add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(
            -1)]
        bias = bias.view(self.wh*self.ww, self.wh*self.ww, -
                         1).permute(2, 0, 1)  # [h, N, N]
        attn = attn + bias.unsqueeze(0)  # broadcast to Bn

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)  # [Bn, h, N, N]

        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(Bn, N, C)
        return self.proj(out)

# ----- SWIN BLOCK -----


class SwinBlock2D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.dim = dim
        self.wh, self.ww = window_size
        self.sh, self.sw = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def _build_attn_mask(self, H, W, device):
        # build mask only when shifted
        if self.sh == 0 and self.sw == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)  # [1,H,W,1]
        cnt = 0
        for h in (slice(0, -self.sh), slice(-self.sh, None)) if self.sh > 0 else (slice(0, H),):
            for w in (slice(0, -self.sw), slice(-self.sw, None)) if self.sw > 0 else (slice(0, W),):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # partition to windows
        mask_windows = window_partition(
            img_mask, self.wh, self.ww).squeeze(-1)  # [nWin, wh*ww]
        nW = mask_windows.shape[0]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask  # [nW, wh*ww, wh*ww]

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, N, C = x.shape
        assert N == H*W

        # save residual before windows/shift
        x_res = x

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.sh or self.sw:
            shifted = torch.roll(x, shifts=(-self.sh, -self.sw), dims=(1, 2))
        else:
            shifted = x

        # window partition
        windows = window_partition(
            shifted, self.wh, self.ww)  # [B*nW, wh*ww, C]
        attn_mask = self._build_attn_mask(H, W, x.device)
        if attn_mask is not None:
            # repeat mask per-batch
            attn_mask = attn_mask.repeat(B, 1, 1)

        # attention
        xw = self.attn(self.norm1(windows), attn_mask=attn_mask)

        # merge windows
        merged = window_reverse(xw, self.wh, self.ww, H, W, B)

        # reverse cyclic shift
        if self.sh or self.sw:
            x_attn = torch.roll(merged, shifts=(self.sh, self.sw), dims=(1, 2))
        else:
            x_attn = merged

        x_attn = x_attn.view(B, H*W, C)

        # add attention residual
        x = x_res + x_attn

        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x

# ----- HEIGHT-ONLY MERGE (2x1) -----


class HeightOnlyPatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, out_dim, kernel_size=(
            2, 1), stride=(2, 1), bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)      # [B,C,H,W]
        x = self.reduce(x)                            # [B,out,H/2,W]
        H2, W2 = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)               # [B,H2*W2,out]
        x = self.norm(x)
        return x, H2, W2

# ----- COMBINING (FC+Act+Drop) -----


class Combining(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, N, C = x.shape
        x = x.view(B, H, W, C).mean(dim=1)  # pool height -> [B,W,C]
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x  # [B, W, out_dim]

# ----- MMS (copied from your HTR-VT) -----


def _mask_random_1d(B, L, ratio, device):
    if ratio <= 0.0:
        return torch.zeros(B, L, dtype=torch.bool, device=device)
    num = int(round(ratio*L))
    noise = torch.rand(B, L, device=device)
    idx = noise.argsort(dim=1)[:, :num]
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


def _mask_block_1d(B, L, ratio, device, min_block=2):
    if ratio <= 0.0:
        return torch.zeros(B, L, dtype=torch.bool, device=device)
    target = int(round(ratio*L))
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        covered = 0
        for _ in range(10000):
            if covered >= target:
                break
            remain = max(1, target - covered)
            blk = random.randint(min_block, min(remain, L))
            start = random.randint(0, max(0, L - blk))
            prev = int(mask[b, start:start+blk].sum().item())
            mask[b, start:start+blk] = True
            covered += blk - prev
    return mask


def _mask_span_1d(self, B: int, L: int, ratio: float, max_span: int, device) -> torch.Tensor:
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


def generate_mms_keep_mask(x, ratios=None, max_span_length=8, block_params=None):
    B, L, _ = x.shape
    if ratios is None:
        ratios = {"random": 0.5, "block": 0.25, "span": 0.25}
    min_block = int((block_params or {}).get("min_block", 2))
    m_rand = _mask_random_1d(B, L, ratios.get("random", 0.5), x.device)
    m_block = _mask_block_1d(B, L, ratios.get(
        "block", 0.25), x.device, min_block)
    m_span = _mask_span_1d(B, L, ratios.get(
        "span", 0.25), max_span_length, x.device)
    m_union = (m_rand | m_block | m_span)
    return (~m_union).float().unsqueeze(-1)  # [B,L,1]


# ----- HYBRID HTR-VT + SWIN -----


class HTR_VT_Swin(nn.Module):
    """
    ResNet (light) -> Swin Stage1 -> H-only merge -> Swin Stage2 -> H-only merge -> Swin Stage3 -> Combine -> Head(CTC)
    Input:  [B, 1, 64, 512]
    CNN out:[B, Cfe, 4, 128]
    Final:  [B, 128, nb_cls]
    """

    def __init__(self,
                 nb_cls=80,
                 d_model=192,            # if None, use Cfe from ResNet; else 1x1 conv to project
                 stage_depths=(2, 2, 2),
                 stage_heads=(4, 6, 8),
                 stage_windows=((4, 8), (2, 8), (1, 8)),
                 stage_shifts=((0, 0), (0, 4), (0, 4)),
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        # CNN feature extractor (already modified)
        # adjust nb_feat to fit your Cfe budget
        self.patch_embed = resnet18.ResNet18(nb_feat=d_model)

        self.proj = None     # set after we see Cfe
        self._cached_shape = None

        # we’ll build Swin blocks after we know Cfe (on first forward) OR
        # initialize with a reasonable guess; then register real modules.
        self._depths = stage_depths
        self._heads = stage_heads
        self._wins = stage_windows
        self._shifts = stage_shifts
        self.mlp_ratio = mlp_ratio
        self.drop = drop

        # Merging & Combining placeholders
        self.merge1 = None
        self.merge2 = None
        self.combiner = None

        # head
        self.head = None  # set after dims are known
        self.nb_cls = nb_cls

        # masking token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1))  # resized later
        nn.init.normal_(self.mask_token, std=0.02)

        # remember requested model dim if provided
        self._requested_d_model = d_model

    def _build_swin(self, Cfe):
        # Get the device of the existing patch_embed to ensure consistency
        device = next(self.patch_embed.parameters()).device
        
        D = Cfe if self._requested_d_model is None else self._requested_d_model
        if D != Cfe:
            self.proj = nn.Conv2d(Cfe, D, kernel_size=1, bias=False).to(device)
        else:
            self.proj = nn.Identity().to(device)

        # Stage 1/2/3 blocks with alternating W-MSA / SW-MSA
        self.stage1 = nn.ModuleList()
        win1 = self._wins[0]
        for i in range(self._depths[0]):
            shift = (0, 0) if i % 2 == 0 else (win1[0]//2, win1[1]//2)
            self.stage1.append(SwinBlock2D(
                D, self._heads[0], win1, shift, self.mlp_ratio, self.drop).to(device))
        self.merge1 = HeightOnlyPatchMerging(D, D*2).to(device)
        D *= 2

        self.stage2 = nn.ModuleList()
        win2 = self._wins[1]
        for i in range(self._depths[1]):
            shift = (0, 0) if i % 2 == 0 else (win2[0]//2, win2[1]//2)
            self.stage2.append(SwinBlock2D(
                D, self._heads[1], win2, shift, self.mlp_ratio, self.drop).to(device))
        self.merge2 = HeightOnlyPatchMerging(D, D*2).to(device)
        D *= 2

        self.stage3 = nn.ModuleList()
        win3 = self._wins[2]
        for i in range(self._depths[2]):
            shift = (0, 0) if i % 2 == 0 else (win3[0]//2, win3[1]//2)
            self.stage3.append(SwinBlock2D(
                D, self._heads[2], win3, shift, self.mlp_ratio, self.drop).to(device))

        self.combiner = Combining(D, D, drop=self.drop).to(device)
        self.head = nn.Linear(D, self.nb_cls).to(device)

        # resize mask_token to correct channel dim and device
        with torch.no_grad():
            self.mask_token = nn.Parameter(torch.zeros(1, 1, D, device=device))
            nn.init.normal_(self.mask_token, std=0.02)

    # ------------- forward_features (with optional MMS masking) -------------
    def forward_features(self, x, use_masking=False, mask_mode="mms",
                         mask_ratio=0.3, max_span_length=8,
                         ratios=None, block_params=None):
        # CNN feature extractor
        # [B,Cfe,4,128] for 64x512 input
        x = self.patch_embed(x)
        B, Cfe, H, W = x.shape  # expect H=4, W=128
        if self._cached_shape is None:
            self._cached_shape = (Cfe, H, W)
            self._build_swin(Cfe)

        # project channels if needed
        x = self.proj(x)                        # [B, D, 4,128]
        D = x.shape[1]

        # flatten to tokens for masking
        x_seq = x.flatten(2).transpose(1, 2)     # [B, H*W, D]

        if use_masking:
            if mask_mode == "mms":
                keep = generate_mms_keep_mask(
                    x_seq, ratios=ratios, max_span_length=max_span_length, block_params=block_params)
            elif mask_mode == "random":
                keep = (~_mask_random_1d(B, H*W, mask_ratio, x.device)
                        ).float().unsqueeze(-1)
            elif mask_mode == "block":
                keep = (~_mask_block_1d(B, H*W, mask_ratio, x.device)
                        ).float().unsqueeze(-1)
            else:  # span
                keep = (~_mask_span_1d(B, H*W, mask_ratio,
                        max_span_length, x.device)).float().unsqueeze(-1)
            x_seq = x_seq * keep + (1 - keep) * self.mask_token  # [B,N,D]

        # back to [B,N,C] for Swin blocks
        x = x_seq

        # Stage 1 @ HxW = 4x128
        for blk in self.stage1:
            x = blk(x, H, W)
        x, H, W = self.merge1(x, H, W)  # -> 2x128, D*=2

        # Stage 2 @ 2x128
        for blk in self.stage2:
            x = blk(x, H, W)
        x, H, W = self.merge2(x, H, W)  # -> 1x128, D*=2

        # Stage 3 @ 1x128 (1D along width windows)
        for blk in self.stage3:
            x = blk(x, H, W)

        # Combine (pool height -> 1) and return features [B,W,D]
        feats = self.combiner(x, H, W)          # [B,128,D]
        return feats

    # ------------- forward -------------
    def forward(self, x, use_masking=False, return_features=False,
                mask_mode="mms", mask_ratio=0.3, max_span_length=8,
                ratios=None, block_params=None):
        feats = self.forward_features(
            x, use_masking, mask_mode, mask_ratio, max_span_length, ratios, block_params)
        logits = self.head(feats)  # [B,128,nb_cls]
        if return_features:
            return logits, feats
        return logits

# Factory to match your create_model signature


def create_model(nb_cls, **kwargs):
    return HTR_VT_Swin(nb_cls=nb_cls, **kwargs)
