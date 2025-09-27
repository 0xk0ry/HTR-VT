# stn_modules.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Version A: Affine STN (simple)
# ------------------------------
class AffineSTN(nn.Module):
    """
    A lightweight STN that predicts a 2x3 affine matrix and rectifies the image.
    - Works well for slant/rotation/scale; not enough for strong curvature.
    """
    def __init__(self, in_ch=1, rectified_size=None):
        """
        rectified_size: (H, W) of the rectified output. If None, keep input size.
        """
        super().__init__()
        self.rectified_size = rectified_size

        # Very small localization CNN
        self.loc = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 6)
        )

        # Identity init (VERY IMPORTANT)
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1,0,0, 0,1,0], dtype=torch.float))

    def forward(self, x):
        B, C, H, W = x.shape
        out_h, out_w = (H, W) if self.rectified_size is None else self.rectified_size

        theta = self.fc(self.loc(x))                         # (B, 6)
        theta = theta.view(-1, 2, 3)                         # (B, 2, 3)

        grid = F.affine_grid(theta, size=(B, C, out_h, out_w), align_corners=False)
        x_warp = F.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                               align_corners=False)
        return x_warp


# ---------------------------------------
# Version B: TPS-STN (for curved text)
# ---------------------------------------
def _build_base_fiducials(K):
    """K control points: K/2 on top edge, K/2 on bottom edge, in normalized [-1,1] coords."""
    assert K % 2 == 0 and K >= 4
    ctrl_pts_top = torch.stack([
        torch.linspace(-1.0, 1.0, steps=K//2),
        torch.full((K//2,), -1.0)
    ], dim=1)
    ctrl_pts_bottom = torch.stack([
        torch.linspace(-1.0, 1.0, steps=K//2),
        torch.full((K//2,), 1.0)
    ], dim=1)
    return torch.cat([ctrl_pts_top, ctrl_pts_bottom], dim=0)    # (K, 2)


def _pairwise_U(r2, eps=1e-6):
    """TPS radial basis function U(r) = r^2 * log(r); we pass r^2 for stability."""
    r = torch.clamp(r2, min=eps).sqrt()
    return r2 * torch.log(r + eps)


class _TPSGridGen(nn.Module):
    """
    Build rectification grid for TPS-STN.
    Precomputes L^{-1} for the base (target) fiducials; at runtime we plug predicted
    source fiducials to get mapping parameters, then generate the sampling grid.
    """
    def __init__(self, out_h, out_w, base_fiducials):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        # base fiducials in normalized [-1,1] coords (K,2)
        self.register_buffer('P', base_fiducials)   # (K,2)
        K = base_fiducials.shape[0]

        # Construct L = [[K, P, 1], [P^T, 0]]
        # K_ij = U(||p_i - p_j||)
        pdist = torch.cdist(base_fiducials, base_fiducials, p=2)  # (K,K)
        K_mat = _pairwise_U(pdist**2)                             # (K,K)
        ones = torch.ones(K, 1, device=base_fiducials.device)
        P_aug = torch.cat([ones, base_fiducials], dim=1)          # (K,3)

        upper = torch.cat([K_mat, P_aug], dim=1)                  # (K, K+3)
        lower = torch.cat([P_aug.transpose(0,1), torch.zeros(3,3, device=base_fiducials.device)], dim=1)  # (3, K+3)
        L = torch.cat([upper, lower], dim=0)                      # (K+3, K+3)
        L_inv = torch.inverse(L)
        self.register_buffer('L_inv', L_inv)                      # (K+3, K+3)

        # Precompute the target grid features [U(||q - p_i||), 1, x, y]
        # IMPORTANT: create all new tensors on the same device as base_fiducials to avoid CPU/CUDA mismatch
        device = base_fiducials.device
        xs = torch.linspace(-1.0, 1.0, out_w, device=device)
        ys = torch.linspace(-1.0, 1.0, out_h, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')    # (H,W)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (HW,2) already on device
        self.register_buffer('Q', grid)                           # (HW,2)

        d = torch.cdist(grid, base_fiducials, p=2)                # (HW,K)
        U = _pairwise_U(d**2)                                     # (HW,K)
        ones_hw = torch.ones(grid.shape[0], 1, device=grid.device)
        self.register_buffer('Phi', torch.cat([U, ones_hw, grid], dim=1))  # (HW, K+3)

    def forward(self, source_fiducials):
        """
        source_fiducials: (B, K, 2) in normalized [-1,1] coordinates on the *input* image.
        Returns a sampling grid for grid_sample: (B, H, W, 2), normalized.
        """
        B, K, _ = source_fiducials.shape
        # Solve for mapping params: [W; A] = L_inv @ [Y; 0]
        Y = torch.cat([source_fiducials, torch.zeros(B, 3, 2, device=source_fiducials.device)], dim=1)  # (B, K+3, 2)
        params = torch.matmul(self.L_inv.unsqueeze(0), Y)  # (B, K+3, 2)

        # Build grid: G = Phi @ params
        G = torch.matmul(self.Phi.unsqueeze(0), params)    # (B, HW, 2)
        G = G.view(B, self.out_h, self.out_w, 2)
        return G


class TPS_STN(nn.Module):
    """
    Thin-Plate Spline STN (RARE/ASTER-style) for curved text lines.
    - Predicts K control points on the *input* image (normalized coords).
    - Warps image so those points align to a fixed, straight set of base fiducials.
    """
    def __init__(self, in_ch=1, K=20, rectified_size=None):
        super().__init__()
        assert K % 2 == 0 and K >= 4
        self.K = K
        self.rectified_size = rectified_size

        # Localization CNN (small & stable)
        self.loc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, K * 2)
        )

        # Identity init to a straight line (VERY IMPORTANT)
        base = _build_base_fiducials(K)        # (K,2) in [-1,1]
        self.register_buffer('base_fid', base)
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(base.view(-1))

        # Grid generator gets built lazily when input size known (if rectified_size is None)
        self._gridgen_cache = None  # (H,W) -> gridgen

    def _get_gridgen(self, device, H, W):
        out_h, out_w = (H, W) if self.rectified_size is None else self.rectified_size
        key = (out_h, out_w)
        if (self._gridgen_cache is None) or (self._gridgen_cache[0] != key):
            gridgen = _TPSGridGen(out_h, out_w, self.base_fid.to(device))
            self._gridgen_cache = (key, gridgen.to(device))
        return self._gridgen_cache[1], out_h, out_w

    def forward(self, x):
        B, C, H, W = x.shape
        gridgen, out_h, out_w = self._get_gridgen(x.device, H, W)

        cp = self.fc(self.loc(x)).view(B, self.K, 2)        # normalized [-1,1]
        cp = torch.tanh(cp)                                 # keep inside image range

        grid = gridgen(cp)                                  # (B, out_h, out_w, 2)
        x_warp = F.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                               align_corners=False)
        return x_warp

# ------------------------------------------------------
# Wrappers that combine STN -> ResNet18 feature extractor
# ------------------------------------------------------
from model import resnet18

class STNOnlyResNet18(nn.Module):
    """Affine STN -> ResNet18 (your feature extractor stays untouched)."""
    def __init__(self, nb_feat=384, in_ch=1, rectified_size=None):
        super().__init__()
        self.stn = AffineSTN(in_ch=in_ch, rectified_size=rectified_size)
        self.backbone = resnet18.ResNet18(nb_feat=nb_feat)

    def forward(self, x):
        # Ensure input is on same device as model parameters
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device, non_blocking=True)
        x = self.stn(x)
        return self.backbone(x)

class TPSSTNResNet18(nn.Module):
    """TPS STN -> ResNet18."""
    def __init__(self, nb_feat=384, in_ch=1, K=20, rectified_size=None):
        super().__init__()
        self.stn = TPS_STN(in_ch=in_ch, K=K, rectified_size=rectified_size)
        self.backbone = resnet18.ResNet18(nb_feat=nb_feat)

    def forward(self, x):
        # Ensure input is on same device as model parameters (prevents X1 CPU / X2 CUDA mismatch)
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device, non_blocking=True)
        x = self.stn(x)
        return self.backbone(x)


# ----------------
# Quick smoke test
# ----------------
if __name__ == "__main__":
    B, C, H, W = 2, 1, 48, 320       # example line crop
    x = torch.randn(B, C, H, W)

    # A) Affine STN
    model_a = STNOnlyResNet18(nb_feat=384, in_ch=1)
    y_a = model_a(x)
    print("Affine STN -> ResNet18 output:", y_a.shape)

    # B) TPS-STN
    model_b = TPSSTNResNet18(nb_feat=384, in_ch=1, K=20)
    y_b = model_b(x)
    print("TPS-STN -> ResNet18 output:", y_b.shape)
