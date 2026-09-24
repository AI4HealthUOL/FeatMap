import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import hashlib
from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms

class LossModule:
    """
    Factory for creating selectable training losses.

    """

    @staticmethod
    def create(name: str, params: dict = None) -> nn.Module:
        params = params or {}
        name = name.lower().strip()

        if "+" in name:
            terms = [
                t.strip()
                for t in name.split("+")
                if t.strip()
            ]

            modules = {}
            weights = params.get("weights", {})

            for term in terms:
                term_params = params.get(term, {})

                modules[term] = LossModule.create(
                    term,
                    term_params,
                )

            return CompositeLoss(
                modules=modules,
                weights=weights,
            )

        if name == "mse":
            return MSELossWrapper()

        if name == "huber":
            return HuberLossWrapper(
                delta=params.get("delta", 1.0)
            )

        if name == "median_cos":
            return MedianCosineLoss()

        if name == "mean_cos":
            return MeanCosineLoss()

        if name == "trimmed_cos":
            return TrimmedMeanCosineLoss(
                trim_frac=params.get("trim_frac", 0.25)
            )

        if name == "patch_mse":
            return PatchMSELoss(
                patch_h=params.get("patch_h", 4),
                patch_w=params.get("patch_w", 4),
            )

        if name == "ntxent":
            return NTXentSpatialLoss(
                temperature=params.get("temp", 0.1),
                sample_neg=params.get("sample_neg", 1024),
            )

        raise ValueError(f"Unknown loss function: {name}")


class MSELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.crit(outputs, targets)


class HuberLossWrapper(nn.Module):
    """
    SmoothL1 / Huber loss wrapper.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()

        try:
            self.crit = nn.SmoothL1Loss(beta=delta)
        except TypeError:
            self.crit = nn.SmoothL1Loss()

        self.delta = delta

    def forward(self, outputs, targets):
        return self.crit(outputs, targets)


class CompositeLoss(nn.Module):
    """
    Weighted combination of named losses.
    """

    def __init__(self, modules: dict, weights: dict = None):
        super().__init__()

        self.loss_modules = nn.ModuleDict(modules)

        self.weights = weights or {}

    def forward(self, outputs, targets):
        total = 0.0

        for name, module in self.loss_modules.items():

            weight = self.weights.get(name, 1.0)

            total = total + weight * module(outputs, targets)

        return total


class MeanCosineLoss(nn.Module):
    """
    1 - mean cosine similarity over spatial tokens.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        B, C, H, W = x.shape

        feat1 = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, H * W, C)
        )

        feat2 = (
            y.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, H * W, C)
        )

        f1 = F.normalize(feat1, p=2.0, dim=2, eps=self.eps)
        f2 = F.normalize(feat2, p=2.0, dim=2, eps=self.eps)

        cos = torch.clamp(
            F.cosine_similarity(f1, f2, dim=2),
            -1 + self.eps,
            1 - self.eps,
        )

        return 1.0 - cos.mean()


class MedianCosineLoss(nn.Module):
    """
    Median cosine similarity loss.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        B, C, H, W = x.shape

        feat1 = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, H * W, C)
        )

        feat2 = (
            y.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, H * W, C)
        )

        f1 = F.normalize(feat1, p=2.0, dim=2, eps=self.eps)
        f2 = F.normalize(feat2, p=2.0, dim=2, eps=self.eps)

        cos = torch.clamp(
            F.cosine_similarity(f1, f2, dim=2),
            -1 + self.eps,
            1 - self.eps,
        )

        sorted_cos, _ = torch.sort(cos, dim=1)

        mid = sorted_cos.shape[1] // 2
        median = sorted_cos[:, mid]

        return 1.0 - median.mean()


class TrimmedMeanCosineLoss(nn.Module):
    """
    Trimmed mean cosine similarity.

    """

    def __init__(self, trim_frac=0.25, eps=1e-8):
        super().__init__()

        assert 0.0 <= trim_frac < 0.5

        self.trim_frac = trim_frac
        self.eps = eps

    def forward(self, x, y):
        B, C, H, W = x.shape

        feat1 = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, H * W, C)
        )

        feat2 = (
            y.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, H * W, C)
        )

        f1 = F.normalize(feat1, p=2.0, dim=2, eps=self.eps)
        f2 = F.normalize(feat2, p=2.0, dim=2, eps=self.eps)

        cos = torch.clamp(
            F.cosine_similarity(f1, f2, dim=2),
            -1 + self.eps,
            1 - self.eps,
        )

        k = int(self.trim_frac * cos.shape[1])

        if k == 0:
            return 1.0 - cos.mean()

        sorted_cos, _ = torch.sort(cos, dim=1)

        trimmed = sorted_cos[:, k:-k]

        return 1.0 - trimmed.mean()

class PatchMSELoss(nn.Module):
    """
    Pool features into patches and compute MSE
    at patch-level.
    """

    def __init__(self, patch_h=4, patch_w=4):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w

        self.crit = nn.MSELoss()

    def forward(self, x, y):
        B, C, H, W = x.shape

        ph = min(self.patch_h, H)
        pw = min(self.patch_w, W)

        x_p = F.adaptive_avg_pool2d(
            x,
            (max(1, H // ph), max(1, W // pw)),
        )

        y_p = F.adaptive_avg_pool2d(
            y,
            (max(1, H // ph), max(1, W // pw)),
        )

        return self.crit(x_p, y_p)


class NTXentSpatialLoss(nn.Module):
    """
    NT-Xent (InfoNCE) loss over spatial tokens.

    Positive pairs:
        matching spatial tokens between output and target

    Negatives:
        sampled tokens from the batch
    """

    def __init__(
        self,
        temperature=0.1,
        sample_neg=1024,
        eps=1e-8,
    ):
        super().__init__()

        self.temperature = temperature
        self.sample_neg = sample_neg
        self.eps = eps

    def forward(self, x, y):
        B, C, H, W = x.shape

        N = H * W

        # (B*N, C)
        fx = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(B * N, C)
        )

        fy = (
            y.permute(0, 2, 3, 1)
            .contiguous()
            .view(B * N, C)
        )

        fx = F.normalize(fx, p=2.0, dim=1, eps=self.eps)
        fy = F.normalize(fy, p=2.0, dim=1, eps=self.eps)

        # Positive logits
        pos = torch.sum(fx * fy, dim=1, keepdim=True)

        # Negative sampling
        idx = torch.randperm(fy.shape[0], device=fy.device)

        neg_idx = idx[: min(self.sample_neg, fy.shape[0])]

        neg_pool = fy[neg_idx]

        # (B*N, K)
        neg_logits = torch.matmul(fx, neg_pool.t())

        logits = torch.cat([pos, neg_logits], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=logits.device,
        )

        loss = F.cross_entropy(logits, labels)

        return loss
        
def process_features(feature, reconstructor):
    gallery = reconstructor.reconstruct_input_img_from_features(feature)
    img = gallery[0]

    if isinstance(img, Image.Image):
        return img

    if isinstance(img, np.ndarray):
        return Image.fromarray(img)

    raise TypeError(f"Unexpected type: {type(img)}")



def process_image(image, reconstructor, extractor_name):
    if extractor_name == "swinv2":
        tfm = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(288),
        ])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = tfm(image)
    tensor = to_tensor(image).unsqueeze(0)

    gallery = reconstructor.reconstruct_input_original(tensor)
    for img in gallery:
        return Image.fromarray(img)
