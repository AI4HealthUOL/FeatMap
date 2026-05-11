import os
import json
import yaml
import time
from PIL import Image

import numpy as np
import torch
import timm

from torch.utils.data import DataLoader
from torchvision import transforms

from prepare_datasets.feature_dataset_impl import (
    PairedFeatureDataset,
    DatasetNormalize,
    test_collate_fn
)

from FeatInv.featinv_reconstructor_conv import InputReconstructorConv
from FeatInv.featinv_reconstructor_swinv2 import InputReconstructorSwinV2

from eval_functions import Evaluator
from eval_helpers import process_features, process_image
from model_implementations import load_model, apply_spatial_transform
from utils import get_img_path_by_id, set_seed, resolve_feature_subdir, get_mapping_run_paths, get_shared_ids, filter_pairs_by_ids, sample_ids

set_seed(42)


"""
Evaluation pipeline for feature-space mapping models.

This script evaluates trained mapping models by:
1. Loading precomputed test feature pairs (original → manipulated)
2. Applying trained mapping models to original features
3. Reconstructing images from mapped features using FeatInv
4. Reconstructing target manipulated features (ground truth)
5. Comparing:
   - mapped reconstruction vs manipulated image
   - mapped reconstruction vs target reconstruction
6. Computing evaluation metrics
"""

def load_pairs_and_meta(extractor_name, root_dir, dataset, dataset_path, feature_subdir, manipulation):
    if dataset == "CUB_200_2011":
        base_root = os.path.join(dataset_path, "CUB_200_2011/CUB_200_2011/")
    elif dataset == "STANFORD_CARS":
        base_root = os.path.join(dataset_path, "STANFORD_CARS/")

    feature_dir = os.path.join(
        base_root,
        f"{extractor_name}_features",
        "augmented_test",
        feature_subdir
    )

    meta_path = os.path.join(feature_dir, "meta.jsonl")

    row_to_meta = {}
    with open(meta_path) as f:
        for line in f:
            m = json.loads(line)
            row_to_meta[m["row_idx"]] = m

    pairs = np.load(os.path.join(feature_dir, "pairs.npy"), allow_pickle=True)
    pairs = pairs[pairs[:, 2] == manipulation]

    return pairs, row_to_meta, feature_dir

def get_backbone(name, device):
    if name == "convnext":
        return timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            features_only=True
        ).eval().to(device)

    return timm.create_model(
        "swinv2_base_window12to24_192to384_22kft1k",
        pretrained=True,
        features_only=False
    ).eval().to(device)


def run_eval_loop(test_loader, step_fn):
    with torch.inference_mode():
        for batch in test_loader:
            step_fn(batch)


def run_spatial_baseline(*args, **kwargs):
    (
        test_loader,
        target_manipulation,
        extractor_eval_path,
        feature_subdir,
        dataset,
        feature_key,
        row_to_meta,
        evaluator,
        reconstructor,
        test_dataset,
        normalized,
        target_recon_paths
    ) = args

    print(">> Running spatial-only baseline")

    def step(batch):
        orig_feats, _, _, _, _, orig_indices, _ = batch
        orig_feats = orig_feats.cuda() if torch.cuda.is_available() else orig_feats

        spatial = apply_spatial_transform(
            orig_feats.clone(), target_manipulation)

        for i, idx in enumerate(orig_indices):

            meta = row_to_meta[idx]
            orig_id = meta["original_id"]

            feat = spatial[i]
            if normalized:
                feat = test_dataset.denormalize(feat)

            img = process_features(feat.cpu().numpy(), reconstructor)

            mode = f"spatial_only_{target_manipulation}"

            save_dir = os.path.join(
                extractor_eval_path,
                f"{mode}/{feature_subdir}/{dataset}/{feature_key}/{target_manipulation}",
            )
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{orig_id}_spatial.png")
            img.save(save_path)

            evaluator.generate_evaluation(
                mode=mode,
                dataset=dataset,
                original_id=orig_id,
                target_manipulation=target_manipulation,
                original_img_path=meta["path"],
                manipulated_img_path=get_img_path_by_id(
                    row_to_meta, orig_id, target_manipulation
                ),
                mapped_img_path=save_path,
                target_reconstructed_img_path=target_recon_paths.get(orig_id),
                eval_save_dir=save_dir,
                feature_key=feature_key,
                showFig=False,
            )

    run_eval_loop(test_loader, step)


def run_mapping_models(*args, **kwargs):
    (
        allowed_models,
        config,
        dataset,
        extractor,
        extractor_model_path,
        normalized,
        feature_subdir,
        target_manipulation,
        device,
        feature_key,
        input_dim,
        output_dim,
        num_feature_vectors,
        test_loader,
        row_to_meta,
        evaluator,
        reconstructor,
        test_dataset,
        target_recon_paths,
        extractor_eval_path,
        apply_flag
    ) = args

    for model_type in allowed_models:

        model_params = config["models"].get(model_type, {})

        model = load_model(
            dataset=dataset,
            extractor_name=extractor,
            model_path=extractor_model_path,
            normalized=normalized,
            feature_subdir=feature_subdir,
            target_manipulation=target_manipulation,
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            model_params=model_params or {},
            device=device,
            feature_key=feature_key,
            num_feature_vectors=num_feature_vectors,
            apply_transform=apply_flag,
        )

        if model is None:
            continue

        def step(batch):
            orig_feats, _, _, _, _, orig_indices, _ = batch
            orig_feats = orig_feats.to(device)
            mapped = model(orig_feats)

            for i, idx in enumerate(orig_indices):

                meta = row_to_meta[idx]
                orig_id = meta["original_id"]

                feat = mapped[i]
                if normalized:
                    feat = test_dataset.denormalize(feat)

                img = process_features(feat.cpu().numpy(), reconstructor)

                mode = f"{model_type}_{target_manipulation}" if apply_flag else model_type

                save_dir = os.path.join(
                    extractor_eval_path,
                    f"{mode}_{normalized}/{feature_subdir}/{dataset}/{feature_key}/{target_manipulation}",
                )
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"{orig_id}_mapped.png")
                img.save(save_path)

                evaluator.generate_evaluation(
                    mode=mode,
                    dataset=dataset,
                    original_id=orig_id,
                    target_manipulation=target_manipulation,
                    original_img_path=meta["path"],
                    manipulated_img_path=get_img_path_by_id(
                        row_to_meta, orig_id, target_manipulation
                    ),
                    mapped_img_path=save_path,
                    target_reconstructed_img_path=target_recon_paths.get(
                        orig_id),
                    eval_save_dir=save_dir,
                    feature_key=feature_key,
                    showFig=False,
                )

        run_eval_loop(test_loader, step)


def main():
    start = time.time()

    with open("../config/test_mapping.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.expandvars(config["model_path"])
    norm_path = os.path.expandvars(config["norm_params_path"])
    dataset_path = os.path.expandvars(config["dataset_path"])
    eval_path = os.path.expandvars(config["evals_path"])

    batch_size = config["batch_size"]
    normalized = config["normalized"]
    subset_size = config.get("subset_size")

    allowed_extractors = config.get("extractor_name", ["convnext", "swinv2"])

    for dataset in config["target_manipulations"]:

        groups = config["target_manipulations"][dataset]

        for group_name, group_cfg in groups.items():

            feature_subdir = resolve_feature_subdir(group_name)
            apply_flag = group_cfg.get("apply_transform", False)
            allowed_models = group_cfg["models"]

            for manipulation in group_cfg["manipulations"]:
                conv_p = conv_m = conv_root = None
                swin_p = swin_m = swin_root = None

                if "convnext" in allowed_extractors:
                    conv_p, conv_m, conv_root = load_pairs_and_meta(
                        "convnext", group_name, dataset, dataset_path, feature_subdir, manipulation
                    )

                if "swinv2" in allowed_extractors:
                    swin_p, swin_m, swin_root = load_pairs_and_meta(
                        "swinv2", group_name, dataset, dataset_path, feature_subdir, manipulation
                    )

                if len(allowed_extractors) == 1:
                    if "convnext" in allowed_extractors:
                        meta_source = conv_m
                    else:
                        meta_source = swin_m

                    all_ids = [m["original_id"] for m in meta_source.values()]
                    selected = sample_ids(all_ids, subset_size or len(all_ids))
                    selected = set(selected)

                    if conv_p is not None:
                        conv_p = filter_pairs_by_ids(conv_p, conv_m, selected)
                    if swin_p is not None:
                        swin_p = filter_pairs_by_ids(swin_p, swin_m, selected)

                else:
                    shared = get_shared_ids(conv_p, conv_m, swin_p, swin_m)
                    selected = sample_ids(shared, subset_size or len(shared))
                    selected = set(selected)

                    conv_p = filter_pairs_by_ids(conv_p, conv_m, selected)
                    swin_p = filter_pairs_by_ids(swin_p, swin_m, selected)

                extractor_data = []

                if "convnext" in allowed_extractors:
                    extractor_data.append(("convnext", conv_p, conv_m, conv_root))

                if "swinv2" in allowed_extractors:
                    extractor_data.append(("swinv2", swin_p, swin_m, swin_root))

                for extractor, pairs, meta, root in extractor_data:

                    evaluator = Evaluator(eval_path, extractor)
                    backbone = get_backbone(extractor, device)

                    extractor_eval_path = os.path.join(eval_path, extractor)
                    os.makedirs(extractor_eval_path, exist_ok=True)

                    feature_cfgs = config["feature_keys"][extractor]

                    for feature_key, feat_cfg in feature_cfgs.items():

                        input_dim = feat_cfg["input_dim"]
                        output_dim = feat_cfg["output_dim"]
                        spatial_size = feat_cfg["spatial_size"]
                        num_vec = spatial_size * spatial_size

                        reconstructor = (
                            InputReconstructorConv(backbone, feature_key)
                            if extractor == "convnext"
                            else InputReconstructorSwinV2(backbone, feature_key)
                        )

                        test_dataset = PairedFeatureDataset(
                            path_prefix=root,
                            feature_key=feature_key,
                            pairs=pairs,
                            manipulation=manipulation
                        )

                        if normalized:
                            test_dataset = DatasetNormalize(
                                test_dataset,
                                dataset,
                                "resized",
                                feature_key,
                                norm_path,
                                False,
                            )

                        target_recon_paths = {}

                        for orig_id in selected:

                            target_img_path = None

                            for m in meta.values():
                                if (
                                    m["original_id"] == orig_id and
                                    m["manipulation"] == manipulation
                                ):
                                    target_img_path = m["path"]
                                    break

                            if target_img_path is None:
                                continue

                            img = Image.open(target_img_path).convert("RGB")

                            target_img = process_image(img, reconstructor, extractor)

                            target_eval_dir = os.path.join(
                                extractor_eval_path,
                                f"{feature_subdir}/{dataset}/{feature_key}/{manipulation}/targets"
                            )
                            os.makedirs(target_eval_dir, exist_ok=True)

                            target_save_path = os.path.join(
                                target_eval_dir,
                                f"{orig_id}_target_recon.png"
                            )

                            target_img.save(target_save_path)
                            target_recon_paths[orig_id] = target_save_path


                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=7,
                            collate_fn=test_collate_fn
                        )
                        if apply_flag:
                            run_spatial_baseline(
                                test_loader,
                                manipulation,
                                extractor_eval_path,
                                feature_subdir,
                                dataset,
                                feature_key,
                                meta,
                                evaluator,
                                reconstructor,
                                test_dataset,
                                normalized,
                                target_recon_paths
                            )    

                        run_mapping_models(
                            allowed_models,
                            config,
                            dataset,
                            extractor,
                            model_path,
                            normalized,
                            feature_subdir,
                            manipulation,
                            device,
                            feature_key,
                            input_dim,
                            output_dim,
                            num_vec,
                            test_loader,
                            meta,
                            evaluator,
                            reconstructor,
                            test_dataset,
                            target_recon_paths,
                            extractor_eval_path,
                            apply_flag
                        )

    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
