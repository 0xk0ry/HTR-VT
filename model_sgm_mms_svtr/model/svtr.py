import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.mask = build_local_mask(H, W, *self.local_k).to(x.device)
            self.mask = self.mask[None, None, :, :]  # [1,1,N,N]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.local:
            attn = attn + self.mask  # restrict attention

        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, N, C)
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


# ----------------------
# SVTR
# ----------------------
class SVTR(nn.Module):
    def __init__(self, num_classes=80, embed_dims=[64, 128, 256], depths=[3, 6, 3], num_heads=[2, 4, 8]):
        super().__init__()
        self.patch_embed = PatchEmbedding(3, embed_dims[0])

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


    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B,N,C]

        for i, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x, H, W)
            if self.mergers[i] is not None:
                x, H, W = self.mergers[i](x, H, W)

        # Combine: pool height → 1
        x = self.combiner(x, H, W)
        logits = self.head(x)
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


def create_model(nb_cls, **kwargs):
    model = SVTR(nb_cls,
                 embed_dims=[64, 128, 256],
                 depths=[3, 6, 3],
                 num_heads=[2, 4, 8])
    return model
