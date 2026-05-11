import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import time
import torch
import os
import json
import re
import yaml
import glob
import numpy as np
from feature_dataset_impl import MemmapFeatureWriter, build_pairs_from_index
from collections import Counter

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Config defines which datasets and train/test splits are extracted
with open("../../config/extract_features.yaml") as f:
    cfg = yaml.safe_load(f)

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

    manip = normalize_manip(manip)

    return manip


class ImageFolderWithPaths(ImageFolder):
    """Store the actual image, target class label, path to the image for later usage."""

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)

        return img, target, path


def forward_swin_last_feature(model, x):
    f = model.forward_features(x)

    # forward_features returns [B, H, W, C]
    f = f.permute(0, 3, 1, 2).contiguous()

    return {0: f}


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
    """Extracts features from batches of images and saves features and their metadata."""
    start = time.time()

    os.makedirs(feature_dir, exist_ok=True)

    imgs, _, _ = next(iter(dataloader))
    imgs = imgs[:2].to(device)

    if model_name == "swinv2_base_window12to24_192to384_22kft1k":
        feats = forward_swin_last_feature(model, imgs)
    else:
        feats = model(imgs)

    fixed_feats = {}
    is_swin = "swin" in model.__class__.__name__.lower()

    for i in target_layers:
        f = feats[i]

        f = f.contiguous()

        fixed_feats[i] = f

        print(f"feat{i} final shape: {f.shape}")

    feature_shapes = {
        f"feat{i}": fixed_feats[i].shape[1:]
        for i in target_layers
    }

    print("Final feature shapes:", feature_shapes)
    writer = MemmapFeatureWriter(
        feature_dir,
        feature_shapes,
        max_samples=len(dataloader.dataset),
    )

    for batch_idx, (images, targets, paths) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        if model_name == "swinv2_base_window12to24_192to384_22kft1k":
            feats = forward_swin_last_feature(model, images)
        else:
            feats = model(images)

        features_dict = {}

        for i in target_layers:
            f = feats[i]

            f = f.contiguous()

            features_dict[f"feat{i}"] = f.detach().cpu()

        short_map = cfg.get("shortened_manipulations", {})

        metadata = []
        for p in paths:
            full_manip = extract_manipulation(p)
            short_manip = short_map.get(full_manip, full_manip)

            metadata.append({
                "original_id": extract_original_id(p),
                "manipulation": short_manip,
                "path": p,
            })

        writer.write_batch(features_dict, targets, paths, metadata)

        if batch_idx % 5 == 0:
            print(batch_idx, "/", len(dataloader))

    index_path = os.path.join(feature_dir, "index.json")
    writer.close(index_path)

    with open(index_path, "r") as f:
        index = json.load(f)

    pairs = build_pairs_from_index(index, orig_key="resized")

    manip_counter = Counter()
    orig_counter = Counter()

    for o_idx, m_idx, manip in pairs:
        manip_counter[manip] += 1
        orig_counter[o_idx] += 1

    print("\n--- Pair count per manipulation ---")

    for m, c in sorted(manip_counter.items()):
        print(f"{m:35s} : {c}")

    print("\n--- Originals usage count ---")

    counts = list(orig_counter.values())

    print("Unique originals:", len(orig_counter))
    print("Min pairs per original:", min(counts))
    print("Max pairs per original:", max(counts))

    pair_path = os.path.join(feature_dir, "pairs.npy")
    np.save(pair_path, pairs, allow_pickle=True)

    print(f"[{split}] pairs built: {len(pairs)}")
    print(f"[{split}] done in {time.time() - start:.2f}s")


def _filter_imagefolder_by_subdir(dataset, subdir_name):
    """Keep only samples that contain subdir_name in their path."""
    if not subdir_name:
        return dataset
    filtered = [(p, y)
                for p, y in dataset.samples if subdir_name in p.split(os.sep)]
    dataset.samples = filtered
    dataset.targets = [s[1] for s in filtered]
    dataset.imgs = filtered
    return dataset


for model_name in cfg["models"]:
    for dataset in datasets:
        if dataset == "CUB_200_2011":
            root_dir = os.path.join(dataset_path, "CUB_200_2011/CUB_200_2011/")
        elif dataset == "STANFORD_CARS":
            root_dir = os.path.join(dataset_path, "STANFORD_CARS/")

        if model_name == "convnext_base.fb_in22k_ft_in1k":
            model = (
                timm.create_model(model_name, pretrained=True,
                                  features_only=True)
                .eval()
                .to(device)
            )
            feature_save_dir_train = "convnext_features/augmented_train"
            feature_save_dir_test = "convnext_features/augmented_test"

            # Resized to the expected input size of the ConvNeXt model
            # Standard Imagenet normalization
            # Based on the same processing from https://github.com/AI4HealthUOL/FeatInv
            transform = transforms.Compose([
                transforms.Resize(288),
                transforms.CenterCrop(288),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        elif model_name == "swinv2_base_window12to24_192to384_22kft1k":
            model = (
                timm.create_model(model_name, pretrained=True,
                                  features_only=False)
                .eval()
                .to(device)
            )
            feature_save_dir_train = "swinv2_features/augmented_train"
            feature_save_dir_test = "swinv2_features/augmented_test"

            transform = transforms.Compose([
                transforms.Resize(384),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        for manipulation_model_dir in manipulation_model_dirs:
            print(
                f"Processing dataset: {dataset}, manipulation model: {manipulation_model_dir}"
            )

            if "train" in extract_from_splits:
                train_image_dir = os.path.join(
                    root_dir, "images/augmented_train")
                train_dataset = ImageFolderWithPaths(
                    root=train_image_dir, transform=transform
                )
                train_dataset = _filter_imagefolder_by_subdir(
                    train_dataset, manipulation_model_dir
                )

                train_feature_base = os.path.join(
                    root_dir, feature_save_dir_train)
                train_feature_dir = os.path.join(
                    train_feature_base, manipulation_model_dir
                )

                os.makedirs(train_feature_dir, exist_ok=True)

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=4,
                    persistent_workers=True
                )

                extract_and_store_features(
                    train_dataloader,
                    train_feature_dir,
                    model_name,
                    dataset,
                    "train",
                    model,
                    target_layers,
                )

            if "test" in extract_from_splits:
                test_image_dir = os.path.join(
                    root_dir, "images/augmented_test")
                test_dataset = ImageFolderWithPaths(
                    root=test_image_dir, transform=transform
                )
                test_dataset = _filter_imagefolder_by_subdir(
                    test_dataset, manipulation_model_dir
                )

                test_feature_base = os.path.join(
                    root_dir, feature_save_dir_test)
                test_feature_dir = os.path.join(
                    test_feature_base, manipulation_model_dir
                )

                os.makedirs(test_feature_dir, exist_ok=True)

                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=4,
                    persistent_workers=True
                )

                extract_and_store_features(
                    test_dataloader,
                    test_feature_dir,
                    model_name,
                    dataset,
                    "test",
                    model,
                    target_layers,
                )
