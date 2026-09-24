import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from prepare_datasets.feature_dataset_impl import (
    MemmapFeatureWriter,
    build_pairs_from_index,
)

from FeatInv.featmap_intermed.swinv2_extractor import (
    SwinV2FeatureExtractor,
)
from FeatInv.featmap_intermed.dinov3_extractor import (
    DINOv3FeatureExtractor,
    DINOV3_MEAN,
    DINOV3_STD,
)

"""
Feature extraction pipeline using ConvNeXt and SwinV2 backbones.

This script converts augmented image datasets into feature-space representations
used throughout the FeatMap pipeline.

Pipeline:
1. Load augmented images (direct + generative manipulations)
2. Apply backbone-specific preprocessing (resize, normalization)
3. Extract intermediate feature maps from selected layers
4. Store features efficiently using memory-mapped arrays (memmap)
5. Save metadata (original_id, manipulation, file path)
6. Build pairing index linking original ↔ manipulated samples

Outputs:
- Feature tensors stored as memmaps (per feature layer)
- index.json with metadata for each sample
- pairs.npy defining training pairs for mapping models

See config/extract_features.yaml for configuration.
"""

SWIN_MODEL_NAME = "swinv2_base_window12to24_192to384_22kft1k"

DINO_MODEL_NAME = "dinov3_vitb16_featmap"
DINO_HF_MODEL_NAME = (
    "facebook/dinov3-vitb16-pretrain-lvd1689m"
)
DINO_IMAGE_SIZE = 224

ALLOWED_MANIPULATIONS = {
    "Add_teddybear",
    "Make_bed_unmade",
    "Color_bedding_blue",
    "Turn_on_lamps",
    "grayscale",
    "hue_shift_60",
    "hue_shift_10",
    "mask_big_square_center",
    "mask_bottom_right_square",
    "mask_small_square_center",
    "mask_top_left_square",
    "noise_40",
    "rotation_90",
    "rotation_180",
    "rotation_270",
    "mirror_h",
    "mirror_v",
    "resized"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Config defines which datasets and train/test splits are extracted


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "extract_features_lsun.yaml"
)

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

datasets = cfg.get("datasets", [cfg.get("dataset")])
model_path = os.path.expandvars(cfg["model_path"])
batch_size = cfg.get("batch_size", 128)
num_workers = cfg.get("num_workers", 6)
pin_memory = cfg.get("pin_memory", True)
target_layers = cfg["target_layers"]
extract_from_splits = cfg["extract_from_splits"]
dataset_path = os.path.expandvars(cfg.get("dataset_path", ""))
manipulation_model_dirs = cfg.get("manipulation_model_dirs", [])


def extract_original_id(path: str):
    """Extracts the unique, original image id from the filename."""
    return os.path.basename(path).split("_")[0]


def normalize_manip(m):
    if isinstance(m, bytes):
        m = m.decode()
    return m


def extract_manipulation(path: str):
    name = os.path.basename(path)
    manip = name.split("_", 1)[1]
    manip = os.path.splitext(manip)[0]
    manip = manip.replace(" ", "_")

    # Normalize qwen prompt variants by dropping the trailing numeric suffix
    # from filenames like ..._0, ..._1, etc.
    if manip.startswith("qwen_"):
        manip = re.sub(r"_\d+$", "", manip)

    manip = normalize_manip(manip)

    return manip

class ImageDatasetWithPaths(Dataset):
    extensions = {".jpg", ".jpeg"}

    def __init__(
        self,
        root,
        transform=None,
        subdir_name=None,
        allowed_manips=None,
        short_map=None,
    ):
        self.transform = transform

        samples = sorted(
            os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk(root)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in self.extensions
        )

        if subdir_name:
            samples = [
                path for path in samples
                if subdir_name in path.split(os.sep)
            ]

        if allowed_manips is not None:
            short_map = short_map or {}
            samples = [
                path
                for path in samples
                if short_map.get(
                    extract_manipulation(path),
                    extract_manipulation(path),
                ) in allowed_manips
            ]

        self.samples = samples
        self.targets = [0] * len(samples)
        self.imgs = list(zip(self.samples, self.targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.targets[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target, path

SWIN_MODEL_NAME = (
    "swinv2_base_window12to24_192to384_22kft1k"
)


def extract_features_for_batch(
    model,
    images,
    model_name,
    target_layers,
):
    if model_name == SWIN_MODEL_NAME:
        return model.extract(
            images,
            target_layers=target_layers,
        )

    if model_name == DINO_MODEL_NAME:
        return model.extract(
            images,
            target_layers=target_layers,
        )

    return model(images)

@torch.inference_mode()
def extract_and_store_features(
    dataloader,
    feature_dir,
    model_name,
    dataset_name,
    split,
    model,
    target_layers,
):
    """Extract and store all configured feature maps."""
    start = time.time()

    os.makedirs(feature_dir, exist_ok=True)

    if len(dataloader.dataset) == 0:
        print(f"[{split}] no images found, skipping")
        return

    try:
        images, _, _ = next(iter(dataloader))
    except StopIteration:
        print(f"[{split}] empty dataloader, skipping")
        return

    images = images[:2].to(
        device,
        non_blocking=True,
    )

    probe_features = extract_features_for_batch(
        model=model,
        images=images,
        model_name=model_name,
        target_layers=target_layers,
    )

    available_layers = set(probe_features.keys())
    requested_layers = set(target_layers)
    missing_layers = requested_layers - available_layers

    if missing_layers:
        raise RuntimeError(
            f"Missing layers {sorted(missing_layers)}. "
            f"Available layers: {sorted(available_layers)}"
        )

    fixed_features = {}

    for layer in target_layers:
        feature = probe_features[layer].contiguous()
        fixed_features[layer] = feature

        print(
            f"feat{layer} final shape: "
            f"{tuple(feature.shape)}"
        )

    feature_shapes = {
        f"feat{layer}": fixed_features[layer].shape[1:]
        for layer in target_layers
    }

    print("Final feature shapes:", feature_shapes)

    writer = MemmapFeatureWriter(
        feature_dir,
        feature_shapes,
        max_samples=len(dataloader.dataset),
    )

    for batch_idx, (
        images,
        targets,
        paths,
    ) in enumerate(dataloader):
        images = images.to(
            device,
            non_blocking=True,
        )

        features = extract_features_for_batch(
            model=model,
            images=images,
            model_name=model_name,
            target_layers=target_layers,
        )

        features_dict = {}

        for layer in target_layers:
            feature = features[layer].contiguous()

            expected_shape = fixed_features[layer].shape[1:]
            actual_shape = feature.shape[1:]

            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Shape mismatch for feat{layer}: "
                    f"expected {tuple(expected_shape)}, "
                    f"got {tuple(actual_shape)}"
                )

            features_dict[f"feat{layer}"] = (
                feature.detach()
                .cpu()
                .contiguous()
            )

        metadata = []

        for path in paths:
            full_manipulation = extract_manipulation(path)
            short_manipulation = short_map.get(
                full_manipulation,
                full_manipulation,
            )

            metadata.append({
                "original_id": extract_original_id(path),
                "manipulation": short_manipulation,
                "path": path,
            })

        writer.write_batch(
            features_dict,
            targets,
            paths,
            metadata,
        )

        if batch_idx % 5 == 0:
            print(
                f"[{split}] {batch_idx} / "
                f"{len(dataloader)}"
            )

    index_path = os.path.join(
        feature_dir,
        "index.json",
    )

    writer.close(index_path)

    with open(index_path, "r", encoding="utf-8") as file:
        index = json.load(file)

    pairs = build_pairs_from_index(
        index,
        orig_key="resized",
    )

    pair_path = os.path.join(
        feature_dir,
        "pairs.npy",
    )

    np.save(
        pair_path,
        pairs,
        allow_pickle=True,
    )

    print(f"[{split}] pairs built: {len(pairs)}")
    print(
        f"[{split}] done in "
        f"{time.time() - start:.2f}s"
    )

def _filter_imagefolder_by_subdir(dataset, subdir_name):
    """Keep only samples that contain subdir_name in their path."""
    if not subdir_name:
        return dataset
    dataset.samples = [
        path for path in dataset.samples if subdir_name in path.split(os.sep)
    ]
    dataset.targets = [0] * len(dataset.samples)
    dataset.imgs = list(zip(dataset.samples, dataset.targets))
    return dataset

def _filter_imagefolder_by_manipulation(dataset, allowed_manips, short_map):
    """Keep only images whose manipulation is in allowed_manips."""

    filtered = []

    for path in dataset.samples:
        manip = extract_manipulation(path)

        if manip.startswith("qwen_"):
            manip = re.sub(r"_\d+$", "", manip)

        manip = short_map.get(manip, manip)

        if manip in allowed_manips:
            filtered.append(path)

    dataset.samples = filtered
    dataset.targets = [0] * len(filtered)
    dataset.imgs = list(zip(dataset.samples, dataset.targets))

    print(f"Kept {len(filtered)} images")

    return dataset

short_map = cfg.get("shortened_manipulations", {})

def make_dataloader(dataset):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs.update({
            "prefetch_factor": 4,
            "persistent_workers": False,
        })

    return DataLoader(
        dataset,
        **loader_kwargs,
    )
def create_model_and_transform(model_name):
    if model_name == "convnext_base.fb_in22k_ft_in1k":
        model = (
            timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,
            )
            .eval()
            .to(device)
        )

        feature_save_dir_train = (
            "convnext_features/augmented_train"
        )
        feature_save_dir_test = (
            "convnext_features/augmented_test"
        )

        transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        return (
            model,
            transform,
            feature_save_dir_train,
            feature_save_dir_test,
        )

    if model_name == SWIN_MODEL_NAME:
        model = SwinV2FeatureExtractor(
            model_name=model_name,
            device=device,
        )

        feature_save_dir_train = (
            "swinv2_features/augmented_train"
        )
        feature_save_dir_test = (
            "swinv2_features/augmented_test"
        )

        transform = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        return (
            model,
            transform,
            feature_save_dir_train,
            feature_save_dir_test,
        )

    if model_name == DINO_MODEL_NAME:
        model = DINOv3FeatureExtractor(
            model_name=DINO_HF_MODEL_NAME,
            hf_token=os.environ.get("HF_TOKEN"),
            device=device,
        )

        feature_save_dir_train = (
            "dinov3_features/augmented_train"
        )
        feature_save_dir_test = (
            "dinov3_features/augmented_test"
        )

        transform = transforms.Compose([
            transforms.Resize(DINO_IMAGE_SIZE),
            transforms.CenterCrop(DINO_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DINOV3_MEAN,
                std=DINOV3_STD,
            ),
        ])

        return (
            model,
            transform,
            feature_save_dir_train,
            feature_save_dir_test,
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )

def get_target_layers(config, model_name):
    """
    Supports either:

    target_layers: [0, 1, 2, 3]

    or:

    target_layers:
      model_name_a: [0, 1, 2, 3]
      model_name_b: [0, 1, 2, 3]
    """
    configured_layers = config["target_layers"]

    if isinstance(configured_layers, dict):
        if model_name not in configured_layers:
            raise KeyError(
                f"No target_layers configured for {model_name}"
            )
        configured_layers = configured_layers[model_name]

    return list(configured_layers)

for model_name in cfg["models"]:
    model_target_layers = get_target_layers(
        cfg,
        model_name,
    )

    for dataset in datasets:
        root_dir = os.path.join(
            dataset_path,
            dataset,
        )

        (
            model,
            transform,
            feature_save_dir_train,
            feature_save_dir_test,
        ) = create_model_and_transform(model_name)

        for manipulation_model_dir in manipulation_model_dirs:
            print(
                f"Processing dataset: {dataset}, "
                f"manipulation model: "
                f"{manipulation_model_dir}"
            )

            if "train" in extract_from_splits:
                train_image_dir = os.path.join(
                    root_dir,
                    "images",
                    "augmented_train",
                )

                train_dataset = ImageDatasetWithPaths(
                    root=train_image_dir,
                    transform=transform,
                )

                train_dataset = (
                    _filter_imagefolder_by_subdir(
                        train_dataset,
                        manipulation_model_dir,
                    )
                )

                train_dataset = (
                    _filter_imagefolder_by_manipulation(
                        train_dataset,
                        ALLOWED_MANIPULATIONS,
                        short_map,
                    )
                )

                train_feature_dir = os.path.join(
                    root_dir,
                    feature_save_dir_train,
                    manipulation_model_dir,
                )

                train_dataloader = make_dataloader(
                    train_dataset,
                )

                extract_and_store_features(
                    dataloader=train_dataloader,
                    feature_dir=train_feature_dir,
                    model_name=model_name,
                    dataset_name=dataset,
                    split="train",
                    model=model,
                    target_layers=model_target_layers,
                )

            if "test" in extract_from_splits:
                test_image_dir = os.path.join(
                    root_dir,
                    "images",
                    "augmented_test",
                )

                test_dataset = ImageDatasetWithPaths(
                    root=test_image_dir,
                    transform=transform,
                )

                test_dataset = (
                    _filter_imagefolder_by_subdir(
                        test_dataset,
                        manipulation_model_dir,
                    )
                )

                test_dataset = (
                    _filter_imagefolder_by_manipulation(
                        test_dataset,
                        ALLOWED_MANIPULATIONS,
                        short_map,
                    )
                )

                test_feature_dir = os.path.join(
                    root_dir,
                    feature_save_dir_test,
                    manipulation_model_dir,
                )

                test_dataloader = make_dataloader(
                    test_dataset,
                )

                extract_and_store_features(
                    dataloader=test_dataloader,
                    feature_dir=test_feature_dir,
                    model_name=model_name,
                    dataset_name=dataset,
                    split="test",
                    model=model,
                    target_layers=model_target_layers,
                )

        if hasattr(model, "close"):
            model.close()