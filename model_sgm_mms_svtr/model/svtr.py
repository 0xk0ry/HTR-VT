%%writefile /kaggle/working/HTR-VT/model_sgm_mms_svtr/model/svtr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ----------------------
# Patch Embedding
# ----------------------


class PatchEmbedding(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, embed_dim // 2,
                               kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim,
                               kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x   # [B, D, H/4, W/4]


# ----------------------
# Local Mask Builder
# ----------------------
def build_local_mask(H, W, hk=7, wk=11):
    mask = torch.ones(H * W,
                      H + hk - 1,
                      W + wk - 1,
                      dtype=torch.float32)
    for h in range(H):
        for w in range(W):
            mask[h * W + w, h:h + hk, w:w + wk] = 0.0
    mask = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
    mask[mask >= 1] = -float("inf")
    return mask  # [H*W, H*W]


# ----------------------
# Attention
# ----------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, local=False, local_k=(7, 11)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.local = local
        self.local_k = local_k
        self.mask = None

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.local and (self.mask is None or self.mask.shape[-1] != N):
            # Create local mask based on actual sequence length N
            if H * W == N:
                # Standard case: sequence length matches spatial dimensions
                self.mask = build_local_mask(H, W, *self.local_k).to(x.device)
                self.mask = self.mask[None, None, :, :]  # [1,1,N,N]
            else:
                # Fallback: create a simpler local mask for mismatched dimensions
                # Use a sliding window approach
                mask = torch.zeros(N, N, device=x.device)
                window_size = min(self.local_k[1], N)  # use width kernel size
                for i in range(N):
                    start = max(0, i - window_size // 2)
                    end = min(N, i + window_size // 2 + 1)
                    # 0 means attend, will be negated below
                    mask[i, start:end] = 0.0
                mask = mask.masked_fill(mask == 1, -float('inf'))
                self.mask = mask[None, None, :, :]  # [1,1,N,N]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = q.transpose(1, 2), k.transpose(
            1, 2), v.transpose(1, 2)  # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        if self.local and self.mask is not None:
            # Ensure mask has the right shape [1, 1, N, N] -> [B, num_heads, N, N]
            mask = self.mask.expand(B, self.num_heads, N, N)
            attn = attn + mask  # restrict attention

        attn = attn.softmax(dim=-1)
        # [B, num_heads, N, head_dim] -> [B, N, C]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ----------------------
# Mixing Block
# ----------------------
class MixingBlock(nn.Module):
    def __init__(self, dim, num_heads=8, local=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, local=local)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class Combining(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C).mean(dim=1)  # pool height -> 1
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x  # [B, W, out_dim]

# ----------------------
# Merging
# ----------------------


class Merging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=(2, 1), padding=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]
        x = self.conv(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.norm(x)
        return x, H, W

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
# ----------------------
# SVTR
# ----------------------


class SVTR(nn.Module):
    def __init__(self, num_classes=80, embed_dims=[64, 128, 256], depths=[3, 6, 3], num_heads=[2, 4, 8], in_channels=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dims[0])

        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()

        for i in range(len(embed_dims)):
            blocks = []
            for j in range(depths[i]):
                # [L]6[G]6 ordering → local first, then global
                local = (j < depths[i] // 2)
                blocks.append(MixingBlock(
                    embed_dims[i], num_heads[i], local=local))
            self.stages.append(nn.ModuleList(blocks))
            if i < len(embed_dims)-1:
                self.mergers.append(Merging(embed_dims[i], embed_dims[i+1]))
            else:
                self.mergers.append(None)

        self.combiner = Combining(embed_dims[-1], embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        # mask_token should match the dimension after patch embedding (first stage)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        nn.init.normal_(self.mask_token, std=.02)
        
        # Anti-blank collapse mechanisms
        self.anti_blank_bias = 3.0  # Strong bias against blank predictions
        self.gradient_clip_val = 1.0  # Gradient clipping for stability
        self.blank_penalty_weight = 0.5  # Weight for blank penalty loss
        
        # Improved initialization to prevent collapse
        with torch.no_grad():
            # Initialize head bias to favor non-blank classes
            if self.head.bias is not None:
                self.head.bias.data[0] = -self.anti_blank_bias  # Bias against blank (class 0)
                self.head.bias.data[1:] = 0.1  # Small positive bias for non-blank classes

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ----------------------
    # masking helpers (copied from HTR-VT)
    # ----------------------
    def _mask_random_1d(self, B, L, ratio, device):
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        num = int(round(ratio * L))
        noise = torch.rand(B, L, device=device)
        idx = noise.argsort(dim=1)[:, :num]
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return mask

    def _mask_block_1d(self, B, L, ratio, device, min_block=2):
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        target = int(round(ratio * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            covered = 0
            for _ in range(10000):
                if covered >= target:
                    break
                remain = max(1, target - covered)
                # Ensure valid range for random.randint
                max_blk = min(remain, L)
                if max_blk < min_block:
                    blk = max_blk  # Use remaining size if it's smaller than min_block
                else:
                    blk = random.randint(min_block, max_blk)

                if blk <= 0:
                    break

                start = random.randint(0, max(0, L - blk))
                prev = int(mask[b, start:start+blk].sum().item())
                mask[b, start:start+blk] = True
                covered += blk - prev
        return mask

    def _mask_span_old_1d(self, B, L, ratio, max_span, device):
        if ratio <= 0.0 or max_span <= 0 or L <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        span_total = int(L * ratio)
        num_spans = max(1, span_total // max_span)
        s = min(max_span, L)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for _ in range(num_spans):
            start = torch.randint(0, L - s + 1, (1,), device=device).item()
            mask[:, start:start+s] = True
        return mask

    def _mask_span_1d(self, B, L, ratio, max_span, device):
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        target = int(round(ratio * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            covered = 0
            for _ in range(10000):
                if covered >= target:
                    break
                s = random.randint(1, max_span)
                l = random.randint(0, L - s)
                r = l + s
                if not mask[b, l:r].any():
                    mask[b, l:r] = True
                    covered += s
        return mask

    def generate_mms_mask(self, x, ratios=None, max_span_length=8, block_params=None):
        B, L, _ = x.shape
        if ratios is None:
            ratios = {"random": 0.5, "block": 0.25, "span": 0.25}
        min_block = int((block_params or {}).get("min_block", 2))
        m_rand = self._mask_random_1d(
            B, L, ratios.get("random", 0.5), x.device)
        m_block = self._mask_block_1d(
            B, L, ratios.get("block", 0.25), x.device, min_block)
        m_span = self._mask_span_1d(B, L, ratios.get(
            "span", 0.25), max_span_length, x.device)
        m_union = (m_rand | m_block | m_span)
        return (~m_union).float().unsqueeze(-1)  # [B,L,1]

    def apply_masking(self, x, use_masking=False, mask_mode="mms",
                      mask_ratio=0.5, max_span_length=8, ratios=None, block_params=None):
        if not use_masking:
            return x
        B, N, D = x.shape
        if mask_mode == "random":
            keep = (~self._mask_random_1d(
                B, N, mask_ratio, x.device)).float().unsqueeze(-1)
        elif mask_mode == "block":
            keep = (~self._mask_block_1d(
                B, N, mask_ratio, x.device)).float().unsqueeze(-1)
        elif mask_mode == "span_old":
            keep = (~self._mask_span_old_1d(B, N, mask_ratio,
                    max_span_length, x.device)).float().unsqueeze(-1)
        else:  # mms
            keep = self.generate_mms_mask(
                x, ratios=ratios, max_span_length=max_span_length, block_params=block_params)
        return x * keep + (1 - keep) * self.mask_token

    # ----------------------
    # forward_features & forward
    # ----------------------
    def forward_features(self, x, use_masking=False, mask_mode="mms",
                         mask_ratio=0.5, max_span_length=8,
                         ratios=None, block_params=None):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B,N,C]
        x = self.apply_masking(x, use_masking, mask_mode, mask_ratio,
                               max_span_length, ratios, block_params)
        for i, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x, H, W)
            if self.mergers[i] is not None:
                x, H, W = self.mergers[i](x, H, W)
        feats = self.combiner(x, H, W)   # [B,W,C]
        return feats

    def forward(self, x, use_masking=False, return_features=False,
                mask_mode="mms", mask_ratio=0.5, max_span_length=8,
                ratios=None, block_params=None):
        # Ensure masking is disabled during evaluation
        if not self.training:
            use_masking = False

        feats = self.forward_features(x, use_masking, mask_mode,
                                      mask_ratio, max_span_length,
                                      ratios, block_params)  # [B,W,C]
        logits = self.head(feats)  # [B,W,num_classes]
        
        # Apply enhanced anti-blank bias and regularization
        if hasattr(self, 'anti_blank_bias') and self.anti_blank_bias > 0:
            # Stronger bias against blank predictions
            logits[:, :, 0] -= self.anti_blank_bias
            
            # Add temperature scaling during training to prevent overconfident predictions
            if self.training:
                temperature = 1.2
                logits = logits / temperature
                
                # Monitor and penalize excessive blank predictions
                probs = torch.softmax(logits, dim=-1)
                blank_ratio = probs[:, :, 0].mean()
                
                # If too many blanks, add small noise to encourage diversity
                if blank_ratio > 0.7:  # If more than 70% blank predictions
                    # Add small regularization noise to non-blank classes
                    noise_scale = 0.1 * self.blank_penalty_weight
                    logits[:, :, 1:] += noise_scale * torch.randn_like(logits[:, :, 1:]) * 0.01

        # Debug printing during evaluation (only for first few samples)
        if not self.training and not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        
        if not self.training and self._debug_counter < 3:
            self._debug_counter += 1
            print(f"\n=== DEBUG SAMPLE {self._debug_counter} ===")
            print(f"Input shape: {x.shape}")
            print(f"Features shape: {feats.shape}")
            print(f"Logits shape: {logits.shape}")
            print(f"Logits stats - min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")
            
            # Show raw predictions (argmax)
            preds = logits.argmax(dim=-1)  # [B, W]
            print(f"Raw predictions shape: {preds.shape}")
            print(f"Raw predictions (first 20): {preds[0][:20].tolist()}")
            
            # Show probability distribution for first few positions
            probs = torch.softmax(logits, dim=-1)
            print(f"Probs for pos 0: top5 = {torch.topk(probs[0, 0], 5)}")
            print(f"Probs for pos 10: top5 = {torch.topk(probs[0, 10], 5)}")
            
            # Check if all predictions are the same (sign of a problem)
            unique_preds = torch.unique(preds[0])
            print(f"Unique predictions: {len(unique_preds)} values = {unique_preds[:10].tolist()}")
            print("=== END DEBUG ===\n")

        if return_features:
            return logits, feats
        return logits


def svtr_tiny(num_classes=80):
    return SVTR(
        num_classes=num_classes,
        embed_dims=[64, 128, 256],
        depths=[3, 6, 3],
        num_heads=[2, 4, 8],
    )


def svtr_small(num_classes=80):
    return SVTR(
        num_classes=num_classes,
        embed_dims=[96, 192, 256],
        depths=[3, 6, 6],
        num_heads=[3, 6, 8],
    )


def svtr_base(num_classes=80):
    return SVTR(
        num_classes=num_classes,
        embed_dims=[128, 256, 384],
        depths=[3, 6, 9],
        num_heads=[4, 8, 12],
    )


def svtr_large(num_classes=80):
    return SVTR(
        num_classes=num_classes,
        embed_dims=[192, 256, 512],
        depths=[3, 9, 9],
        num_heads=[6, 8, 16],
    )


def create_model(nb_cls, in_channels=1, **kwargs):
    model = SVTR(nb_cls,
                 embed_dims=[64, 128, 256],
                 depths=[3, 6, 3],
                 num_heads=[2, 4, 8],
                 in_channels=in_channels)
    return model
