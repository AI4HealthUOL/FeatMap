import json
import os
import cv2
from PIL import Image
from torchvision.transforms import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import re
import timm
from skimage.metrics import structural_similarity, mean_squared_error
import lpips
import numpy as np
import matplotlib.gridspec as gridspec
import csv


def safe_filename(s):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(s))


class Evaluator:
    """
    Evaluator for image reconstruction quality.

    This class compares:
    - manipulated ground truth images
    - model-mapped reconstructions
    - optionally reconstructed targets

    It computes:
    1. Pixel-level metrics (MSE, SSIM)
    2. Perceptual similarity (LPIPS)
    3. Feature-space similarity (Median CosSIM)

    """

    def __init__(self, eval_save_path, extractor_name):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
        self.lpips_model.eval()

        self.feature_extraction_model = (
            timm.create_model(
                "convnext_base.fb_in22k_ft_in1k", pretrained=True, features_only=True
            )
            .eval()
            .to(self.device)
        )

        self.csv_path = os.path.join(
            eval_save_path, f"evaluations_{extractor_name}.csv")

        os.makedirs(eval_save_path, exist_ok=True)

        self.tf = transforms.Compose(
            [
                transforms.Resize((288, 288)),
                transforms.ToTensor(),
            ]
        )

    def log_evaluation(
        self,
        mode,
        dataset,
        feature_key,
        original_id,
        target_manipulation,
        original_img_path,
        manipulated_img_path,
        mapped_img_path,
        fullpath,
        target_vs_mapped_metrics,
        target_reconstructed_vs_mapped_metrics,
    ):
        fieldnames = [
            "mode",
            "dataset",
            "feature_key",
            "original_id",
            "target_manipulation",
            "original_img_path",
            "manipulated_img_path",
            "mapped_img_path",
            "eval_fig_path",
            "target_vs_mapped_MSE",
            "target_vs_mapped_SSIM",
            "target_vs_mapped_LPIPS",
            "target_vs_mapped_MEDIAN_COS_SIM",
            "target_vs_mapped_MASKED_MEDIAN_COS_SIM",
            "target_reconstructed_vs_mapped_MSE",
            "target_reconstructed_vs_mapped_SSIM",
            "target_reconstructed_vs_mapped_LPIPS",
            "target_reconstructed_vs_mapped_MEDIAN_COS_SIM",
            "target_reconstructed_vs_mapped_MASKED_MEDIAN_COS_SIM",
        ]
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "mode": mode,
                    "dataset": dataset,
                    "feature_key": feature_key,
                    "original_id": original_id,
                    "target_manipulation": target_manipulation,
                    "original_img_path": original_img_path,
                    "manipulated_img_path": manipulated_img_path,
                    "mapped_img_path": mapped_img_path,
                    "eval_fig_path": fullpath,
                    "target_vs_mapped_MSE": target_vs_mapped_metrics["MSE"],
                    "target_vs_mapped_SSIM": target_vs_mapped_metrics["SSIM"],
                    "target_vs_mapped_LPIPS": target_vs_mapped_metrics["LPIPS"],
                    "target_vs_mapped_MEDIAN_COS_SIM": target_vs_mapped_metrics[
                        "median_cos_sim"
                    ],
                    "target_vs_mapped_MASKED_MEDIAN_COS_SIM": target_vs_mapped_metrics[
                        "masked_median_cos_sim"
                    ],
                    "target_reconstructed_vs_mapped_MSE": target_reconstructed_vs_mapped_metrics[
                        "MSE"
                    ],
                    "target_reconstructed_vs_mapped_SSIM": target_reconstructed_vs_mapped_metrics[
                        "SSIM"
                    ],
                    "target_reconstructed_vs_mapped_LPIPS": target_reconstructed_vs_mapped_metrics[
                        "LPIPS"
                    ],
                    "target_reconstructed_vs_mapped_MEDIAN_COS_SIM": target_reconstructed_vs_mapped_metrics[
                        "median_cos_sim"
                    ],
                    "target_reconstructed_vs_mapped_MASKED_MEDIAN_COS_SIM": target_reconstructed_vs_mapped_metrics[
                        "masked_median_cos_sim"
                    ],
                }
            )

    def get_lpips_eval(self, train_img, reconstructed_img):
        def to_tensor(img):
            tensor = torch.from_numpy(img).permute(
                2, 0, 1).float() / 127.5 - 1.0
            return tensor.unsqueeze(0).to(self.device)

        x = to_tensor(train_img)
        y = to_tensor(reconstructed_img)

        with torch.no_grad():
            score = self.lpips_model(x, y)

        return score.item()

    def extract_features(self, img, model, target_layers, device):
        transform = transforms.Compose(
            [
                transforms.Resize(288),
                transforms.CenterCrop(288),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_tensor = transform(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            features = model(img_tensor)
            features_dict = {
                f"feat{idx}": features[idx].cpu() for idx in target_layers}
        return features_dict["feat3"]

    def quantile_cosine_similarity_per_superpixel(self, f1, f2, quantile=0.05):
        """
        Quantile cosine similarity over all superpixels (H*W).
        """
        B, C, H, W = f1.shape
        feat1_flat = f1.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat2_flat = f2.permute(0, 2, 3, 1).reshape(B, H * W, C)
        cos_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=2)
        quantile_per_sample = cos_sim.quantile(quantile, dim=1)
        return quantile_per_sample.mean().item()

    def min_cosine_similarity_per_superpixel(self, f1, f2):
        """
        Minimum cosine similarity over all superpixels (H*W)
        """
        B, C, H, W = f1.shape
        feat1_flat = f1.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat2_flat = f2.permute(0, 2, 3, 1).reshape(B, H * W, C)
        cos_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=2)
        min_per_sample = cos_sim.min(dim=1).values
        return min_per_sample.mean().item()

    def calc_metrics(self, img1, img2, mask=None):
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        lpips_score = self.get_lpips_eval(img1_np, img2_np)
        mse_result = mean_squared_error(img1_np.flatten(), img2_np.flatten())
        ssim_result = structural_similarity(
            img1_np, img2_np, channel_axis=-1, data_range=255
        )

        f1 = self.extract_features(
            img1, self.feature_extraction_model, [3], self.device
        )
        f2 = self.extract_features(
            img2, self.feature_extraction_model, [3], self.device
        )

        median_cos_sim = self.median_cosine_similarity_per_superpixel(f1, f2)
        masked_median_cos_sim = None
        if mask is not None:
            masked_median_cos_sim = self.median_cosine_similarity_per_superpixel(
                f1, f2, mask
            )

        results = {
            "MSE": mse_result,
            "SSIM": ssim_result,
            "LPIPS": lpips_score,
            "median_cos_sim": median_cos_sim,
            "masked_median_cos_sim": masked_median_cos_sim,
        }

        return results

    def get_mask(self, original_np, target_np):
        original_img = original_np[..., ::-1]
        target_img = target_np[..., ::-1]

        target_img = cv2.resize(
            target_img, (original_img.shape[1], original_img.shape[0])
        )

        # get euclidean distances between points
        diff = cv2.absdiff(
            cv2.GaussianBlur(original_img, (5, 5), 0),
            cv2.GaussianBlur(target_img, (5, 5), 0),
        ).astype(np.float32)
        diff_norm = np.linalg.norm(diff, axis=2)
        # threshold to create binary mask
        full_mask = (diff_norm > 50).astype(np.uint8)

        # downsample with maxpool to get feat dim mask
        h, w = full_mask.shape
        spatial_dim = 9
        cell_h, cell_w = h // spatial_dim, w // spatial_dim

        mask_small = np.zeros((spatial_dim, spatial_dim), np.uint8)
        for i in range(spatial_dim):
            for j in range(spatial_dim):
                y1, y2 = i * cell_h, min((i + 1) * cell_h, h)
                x1, x2 = j * cell_w, min((j + 1) * cell_w, w)
                mask_small[i, j] = 1 if np.any(full_mask[y1:y2, x1:x2]) else 0

        return mask_small * 255

    def median_cosine_similarity_per_superpixel(self, f1, f2, mask=None):
        """
        Median cosine similarity over all superpixels (H*W)
        """

        B, C, H, W = f1.shape

        feat1_flat = f1.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat2_flat = f2.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if mask is not None:
            mask_bool = mask.reshape(-1) > 0
            idx = np.nonzero(mask_bool)[0]
            if len(idx) == 0:
                return 0.0
            idx = torch.from_numpy(idx).long().to(feat1_flat.device)

            feat1_sel = feat1_flat[:, idx, :]
            feat2_sel = feat2_flat[:, idx, :]
            cos_sim = F.cosine_similarity(feat1_sel, feat2_sel, dim=2)
        else:
            if H * W == 0:
                return 0.0
            cos_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=2)

        median_per_sample = cos_sim.median(dim=1).values
        return median_per_sample.mean().item()

    def generate_evaluation(
        self,
        mode,
        dataset,
        original_id,
        target_manipulation,
        original_img_path,
        manipulated_img_path,
        mapped_img_path,
        target_reconstructed_img_path,
        eval_save_dir,
        feature_key,
        showFig,
    ):
        def load_if_path(img):
            if isinstance(img, str):
                return Image.open(img).convert("RGB")
            return img.convert("RGB") if img.mode != "RGB" else img

        original_img = load_if_path(original_img_path)
        manipulated_img = load_if_path(manipulated_img_path)
        mapped_img = load_if_path(mapped_img_path)
        mapped_img = mapped_img.resize(manipulated_img.size)
        target_reconstructed_img = load_if_path(target_reconstructed_img_path)
        target_reconstructed_img = target_reconstructed_img.resize(
            manipulated_img.size)

        original_np = np.array(original_img)
        manipulated_np = np.array(manipulated_img)

        # masking to select feature vectors that are connected
        # to the changed places in the image
        mask = self.get_mask(original_np, manipulated_np)

        target_vs_mapped_metrics = self.calc_metrics(
            manipulated_img, mapped_img, mask=mask
        )
        target_reconstructed_vs_mapped_metrics = self.calc_metrics(
            target_reconstructed_img, mapped_img, mask=mask
        )

        fn = f"{original_id}_{target_manipulation}"
        filename = safe_filename(fn) + ".png"
        fullpath = os.path.join(eval_save_dir, filename)

        self.log_evaluation(
            mode,
            dataset,
            feature_key,
            original_id,
            target_manipulation,
            original_img_path,
            manipulated_img_path,
            mapped_img_path,
            fullpath,
            target_vs_mapped_metrics,
            target_reconstructed_vs_mapped_metrics,
        )
