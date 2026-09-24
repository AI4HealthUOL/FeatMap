import os
import json
import yaml
import time
from PIL import Image
import torchvision.transforms as transforms

import numpy as np
import torch
import timm

from sklearn.decomposition import PCA

from torch.utils.data import DataLoader

from prepare_datasets.feature_dataset_impl import (
    PairedFeatureDataset,
    DatasetNormalize,
    test_collate_fn,
)

from FeatInv.featinv_reconstructor_conv import InputReconstructorConv

from eval_functions import Evaluator
from eval_helpers import process_features, process_image
from model_implementations import load_model, apply_spatial_transform
from utils import (
    get_img_path_by_id,
    set_seed,
    resolve_feature_subdir,
    filter_pairs_by_ids,
    sample_ids,
)

set_seed(42)

import colorsys
import numpy as np
from PIL import Image


def composition_name(manipulations):
    return "__then__".join(manipulations)


def load_pairs_and_meta(
    extractor_name, dataset, dataset_path, feature_subdir, manipulation
):
    base_root = os.path.join(dataset_path, dataset)

    feature_dir = os.path.join(
        base_root, f"{extractor_name}_features", "augmented_test", feature_subdir
    )

    meta_path = os.path.join(feature_dir, "meta.jsonl")

    row_to_meta = {}
    orig_manip_to_path = {}

    with open(meta_path) as f:
        for line in f:
            m = json.loads(line)

            row_to_meta[m["row_idx"]] = m

            orig_manip_to_path[
                (m["original_id"], m["manipulation"])
            ] = m["path"]

    pairs = np.load(os.path.join(feature_dir, "pairs.npy"), allow_pickle=True)
    pairs = pairs[pairs[:, 2] == manipulation]

    return pairs, row_to_meta, orig_manip_to_path, feature_dir


def get_backbone(device):
    return (
        timm.create_model(
            "convnext_base.fb_in22k_ft_in1k", pretrained=True, features_only=True
        )
        .eval()
        .to(device)
    )


def run_eval_loop(test_loader, step_fn):
    with torch.inference_mode():
        for batch in test_loader:
            step_fn(batch)
def run_composition_models(args):
    (
        allowed_models,
        config,
        dataset,
        extractor,
        extractor_model_path,
        normalized,
        normalizer,
        composition_name_str,
        manipulations,
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
        extractor_eval_path,
        base_apply_flag,
        loss_name,
    ) = args

    qwen_manips = {
        "Add_teddybear",
        "Make_bed_unmade",
        "Color_bedding_blue",
        "Turn_on_lamps",
    }

    geometric_manips = {
        "rotation_90",
        "rotation_180",
        "rotation_270",
        "mirror_h",
        "mirror_v",
    }

    for model_type in allowed_models:
        model_params = config["models"].get(model_type, {})

        step_models = []
        valid_manips = []

        for manipulation in manipulations:
            feature_subdir = (
                "qwen_gs_1_infsteps_10"
                if manipulation in qwen_manips
                else "direct"
            )

            apply_transform = manipulation in geometric_manips

            model = load_model(
                dataset=dataset,
                extractor_name=extractor,
                model_path=extractor_model_path,
                normalized=normalized,
                feature_subdir=feature_subdir,
                target_manipulation=manipulation,
                model_type=model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                model_params=model_params or {},
                device=device,
                feature_key=feature_key,
                loss_name=loss_name,
                num_feature_vectors=num_feature_vectors,
                apply_transform=apply_transform,
            )

            if model is None:
                print(
                    f"[WARN] missing "
                    f"{model_type}:{feature_subdir}:{manipulation}"
                )
                continue

            step_models.append(model)
            valid_manips.append(manipulation)

        if not step_models:
            print(
                f"[WARN] no valid models for "
                f"{model_type}:{composition_name_str}"
            )
            continue

        def step(batch):
            orig_feats, _, _, _, _, orig_indices, _ = batch

            current_feats = orig_feats.to(device)

            for i, idx in enumerate(orig_indices):
                idx = int(idx)
                meta = row_to_meta[idx]
                orig_id = meta["original_id"]

                save_dir = os.path.join(
                    extractor_eval_path,
                    f"{model_type}_{composition_name_str}_{loss_name}",
                    dataset,
                    feature_key,
                    composition_name_str,
                    str(orig_id),
                )

                os.makedirs(save_dir, exist_ok=True)

                feat = current_feats[i]

                if normalized and normalizer is not None:
                    feat = normalizer.denormalize(feat)

                img = process_features(
                    feat.detach().cpu().numpy(),
                    reconstructor,
                )

                img.save(
                    os.path.join(
                        save_dir,
                        "000_original.png",
                    )
                )

            applied = []

            for step_idx, (manipulation, model) in enumerate(
                zip(valid_manips, step_models),
                start=1,
            ):
                current_feats = model(current_feats)
                applied.append(manipulation)

                name = "__".join(
                    value.replace(" ", "_")
                    for value in applied
                )

                for i, idx in enumerate(orig_indices):
                    idx = int(idx)
                    meta = row_to_meta[idx]
                    orig_id = meta["original_id"]

                    save_dir = os.path.join(
                        extractor_eval_path,
                        f"{model_type}_{composition_name_str}_{loss_name}",
                        dataset,
                        feature_key,
                        composition_name_str,
                        str(orig_id),
                    )

                    os.makedirs(save_dir, exist_ok=True)

                    feat = current_feats[i]

                    if normalized and normalizer is not None:
                        feat = normalizer.denormalize(feat)

                    img = process_features(
                        feat.detach().cpu().numpy(),
                        reconstructor,
                    )

                    img.save(
                        os.path.join(
                            save_dir,
                            f"{step_idx:03d}_{name}.png",
                        )
                    )

        run_eval_loop(test_loader, step)


def main():
    start = time.time()

    with open("../config/test_mapping_lsun_comp.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.expandvars(config["model_path"])
    norm_path = os.path.expandvars(config["norm_params_path"])
    dataset_path = os.path.expandvars(config["dataset_path"])
    eval_path = os.path.expandvars(config["evals_path"])

    batch_size = config["batch_size"]
    subset_size = config.get("subset_size")

    for dataset, comp_groups in config.get(
        "compositions",
        {},
    ).items():
        for comp_name, comp_cfg in comp_groups.items():
            comp_manips = comp_cfg["manipulations"]

            allowed_models = comp_cfg["models"]

            loss_functions = config.get(
                "loss_functions",
                ["mse_median_cos"],
            )

            apply_flag = comp_cfg.get(
                "apply_transform",
                False,
            )

            # The composition input dataset is always loaded from
            # the direct feature directory.
            input_feature_subdir = "direct"

            conv_p, conv_m, orig_path_lookup, conv_root = (
                load_pairs_and_meta(
                    "convnext",
                    dataset,
                    dataset_path,
                    input_feature_subdir,
                    "grayscale",
                )
            )

            all_ids = [
                meta["original_id"]
                for meta in conv_m.values()
            ]

            selected = set(
                sample_ids(
                    all_ids,
                    subset_size or len(all_ids),
                )
            )

            conv_p = filter_pairs_by_ids(
                conv_p,
                conv_m,
                selected,
            )

            if np.asarray(conv_p).ndim != 2:
                raise ValueError(
                    "Filtered input pairs are invalid: "
                    f"shape={np.asarray(conv_p).shape}"
                )

            backbone = get_backbone(device)

            extractor_eval_path = os.path.join(
                eval_path,
                "convnext",
            )
            os.makedirs(
                extractor_eval_path,
                exist_ok=True,
            )

            feature_cfgs = config["feature_keys"]["convnext"]

            for feature_key, feat_cfg in feature_cfgs.items():
                input_dim = feat_cfg["input_dim"]
                output_dim = feat_cfg["output_dim"]
                spatial_size = feat_cfg["spatial_size"]
                normalized = feat_cfg["normalize_features"]

                num_vec = spatial_size * spatial_size

                reconstructor = InputReconstructorConv(
                    backbone,
                    feature_key,
                )

                test_dataset = PairedFeatureDataset(
                    path_prefix=conv_root,
                    feature_key=feature_key,
                    pairs=conv_p,
                    manipulation="grayscale",
                )

                normalizer = None

                if normalized:
                    normalizer = DatasetNormalize(
                        test_dataset,
                        dataset,
                        "resized",
                        feature_key,
                        norm_path,
                        False,
                    )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=7,
                    collate_fn=test_collate_fn,
                )

                evaluator = Evaluator(
                    eval_path,
                    "convnext",
                )

                for loss_name in loss_functions:
                    run_composition_models(
                        (
                            allowed_models,
                            config,
                            dataset,
                            "convnext",
                            model_path,
                            normalized,
                            normalizer,
                            comp_name,
                            comp_manips,
                            device,
                            feature_key,
                            input_dim,
                            output_dim,
                            num_vec,
                            test_loader,
                            conv_m,
                            evaluator,
                            reconstructor,
                            test_dataset,
                            extractor_eval_path,
                            apply_flag,
                            loss_name,
                        )
                    )

    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
