import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import hashlib
from PIL import Image
from torchvision.utils import make_grid


class CombinedLoss(nn.Module):
    """
    Combined loss for training feature to feature mappings

    This loss combines:
    - MSE loss
    - cosine similarity loss over spatial feature maps

    The cosine loss:
    - flattens feature maps into spatial tokens (HxW)
    - computes cosine similarity per spatial location
    - uses the median similarity across locations
    (robust to outliers and localized failures)

    """

    def __init__(self, mse_scale=0.5, cossim_scale=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_scale = mse_scale
        self.cossim_scale = cossim_scale
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        cos_loss = cosine_similarity_loss(outputs, targets)
        return self.mse_scale * mse_loss + self.cossim_scale * cos_loss


def cosine_similarity_loss(x, y, eps=1e-8):
    """Median cosine similarity over each superpixel."""
    B, C, H, W = x.shape
    feat1_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
    feat2_flat = y.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

    feat1_norm = F.normalize(feat1_flat, p=2.0, dim=2, eps=eps)
    feat2_norm = F.normalize(feat2_flat, p=2.0, dim=2, eps=eps)

    cos_sim = torch.clamp(
        F.cosine_similarity(feat1_norm, feat2_norm, dim=2), -1 + eps, 1 - eps
    )

    sorted_cos_sim, _ = torch.sort(cos_sim, dim=1)
    mid_idx = sorted_cos_sim.shape[1] // 2
    median_per_sample = sorted_cos_sim[:, mid_idx]

    return 1.0 - median_per_sample.mean()
