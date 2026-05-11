
from train_mapping import MappingModel
from prepare_datasets.feature_dataset_impl import (
    FeatureDataset,
    PairedFeatureDataset,
    DatasetNormalize,
    test_collate_fn
)
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import json
import yaml
import time
import torch
import numpy as np
import pandas as pd
import sys
import random
import hashlib
from collections import defaultdict

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


sys.path.insert(0, '..')


device = "cuda" if torch.cuda.is_available() else "cpu"


"""Analysis of the weight and bias properties of the trained linear mapping models."""


def resolve_feature_subdir(group_name):
    if group_name.startswith("qwen"):
        return "qwen_gs_1_infsteps_10"
    return "direct"


def get_best_checkpoint(checkpoint_dir, checkpoint_prefix):
    try:
        checkpoint_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith(checkpoint_prefix) and f.endswith(".ckpt")
        ]

        checkpoint_files.sort(
            key=lambda x: (
                int(re.search(r"-v(\d+)", x).group(1))
                if re.search(r"-v(\d+)", x)
                else 0
            ),
            reverse=True,
        )

        if not checkpoint_files:
            return None

        return os.path.join(checkpoint_dir, checkpoint_files[0])
    except FileNotFoundError:
        return None


def load_model(
    dataset,
    model_path,
    normalized,
    feature_subdir,
    target_manipulation,
    model_type,
    input_dim,
    output_dim,
    model_params,
    device,
    feature_key,
    num_feature_vectors=None,
    apply_transform=False,
):
    """Load the trained mapping model from its checkpoint"""
    spatial_size = int(num_feature_vectors**0.5) if num_feature_vectors else 9

    model_config = {
        "model_type": model_type,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "spatial_size": spatial_size,
        "manipulation": target_manipulation,
        "apply_transform": apply_transform,
        **(model_params or {}),
    }

    model = MappingModel(model_config).to(device)

    if apply_transform:
        model_folder_name = f"{model_type}_{target_manipulation}"
        checkpoint_dir = os.path.join(
            model_path,
            feature_subdir,
            dataset,
            model_folder_name
        )
    else:
        model_folder_name = model_type
        checkpoint_dir = os.path.join(
            model_path,
            feature_subdir,
            dataset,
            model_folder_name,
            target_manipulation
        )

    checkpoint_prefix = f"{model_type}_{feature_key}_normalized_{normalized}_mapping_model"

    model_checkpoint_path = get_best_checkpoint(
        checkpoint_dir, checkpoint_prefix
    )

    print(f"Loading mapping checkpoint: {model_checkpoint_path}")

    if model_checkpoint_path is None:
        return None

    state = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    return model


def compute_metrics(
    model,
    loader,
    spatial_size,
    input_dim,
    dataset,
    backbone,
    feature_key,
    model_type,
    group_name
):
    """
    Compute metrics for evaluating the Wx and bias properties.

    Metrics computed per sample:
    - wx_norm: median token wise norm of Wx .
    - bias_norm: norm of the bias vector.
    - full_norm: median token wise norm of Wx + bias.
    - bias_effect: median norm of the bias shift (Wx + bias) - Wx.
    - bias_effect_share, wx_share, bias_share: normalized contributions of bias and Wx to full_norm.
    - relative_bias_shift: normalized change in norm when adding bias.
    - cosine_similarity: median cosine similarity between Wx and Wx + bias (directional change).
    """

    rows = []
    model.eval()

    layer = model.model if isinstance(model, MappingModel) else model

    with torch.no_grad():
        for batch in loader:
            orig_feats, _, _, _, manips, orig_indices, target_indices = batch
            orig_feats = orig_feats.to(device)

            B, C, H, W = orig_feats.shape

            # Flatten spatial tokens
            x = orig_feats.permute(0, 2, 3, 1).reshape(B * H * W, C)

            Wx = layer(x)

            bias = (
                layer.bias
                if layer.bias is not None
                else torch.zeros(Wx.shape[-1], device=Wx.device)
            )

            Wx = Wx.reshape(B, H * W, -1)

            y_wx = Wx
            y_full = Wx + bias.view(1, 1, -1)

            eps = 1e-8

            wx_token_norm = y_wx.norm(dim=-1)
            full_token_norm = y_full.norm(dim=-1)

            wx_norm = wx_token_norm.median(dim=1).values
            full_norm = full_token_norm.median(dim=1).values

            bias_norm = bias.norm().expand(B)

            effect = (
                (y_full - y_wx)
                .norm(dim=-1)
                .median(dim=1)
                .values
            )

            wx_share = wx_norm / (full_norm + eps)

            bias_share = bias_norm / (full_norm + eps)

            effect_share = effect / (full_norm + eps)

            dot = (y_wx * y_full).sum(dim=-1)

            cos_sim = dot / (
                wx_token_norm
                * full_token_norm
                + eps
            )

            cos_sim = cos_sim.median(dim=1).values

            input_dominance = (
                wx_norm
                / (full_norm + eps)
            )

            relative_shift = (
                full_norm - wx_norm
            ) / (wx_norm + eps)

            for i in range(B):

                m = manips[i]

                raw_id = (
                    f"{dataset}|{backbone}|{feature_key}|"
                    f"{group_name}|{orig_indices[i]}|{target_indices[i]}"
                )

                sample_id = hashlib.md5(
                    raw_id.encode()
                ).hexdigest()

                base_info = {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "backbone": backbone,
                    "feature": feature_key,
                    "model": f"{model_type}_{m}" if "apply" in group_name else model_type,
                    "group": group_name,
                    "manipulation": m,
                }

                rows.append({
                    **base_info,
                    "metric": "wx_norm",
                    "value": float(wx_norm[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "bias_norm",
                    "value": float(bias_norm[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "full_norm",
                    "value": float(full_norm[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "bias_effect",
                    "value": float(effect[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "bias_effect_share",
                    "value": float(effect_share[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "wx_share",
                    "value": float(wx_share[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "bias_share",
                    "value": float(bias_share[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "cosine_similarity",
                    "value": float(cos_sim[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "input_dominance",
                    "value": float(input_dominance[i]),
                })

                rows.append({
                    **base_info,
                    "metric": "relative_bias_shift",
                    "value": float(relative_shift[i]),
                })

    return rows


manip_groups = {
    "Semantic Editing": [
        "Add_police_lights",
        "Change_body_color_blue",
        "Change_tire_rim_color_red",
        "Remove_side_mirrors",
        "Turn_on_headlights",
    ],
    "Photometric": [
        "grayscale",
        "hue_shift_60",
        "noise_40",
    ],
    "Masking": [
        "mask_big_square_center",
        "mask_bottom_right_square",
        "mask_small_square_center",
        "mask_top_left_square",
    ],
    "Geometry": [
        "rotation_90",
        "rotation_180",
        "rotation_270",
        "mirror_h",
        "mirror_v",
    ],
}


with open("../../config/eval_lin_models.yaml") as f:
    config = yaml.safe_load(f)

dataset_path = os.path.expandvars(
    config["dataset_path"]
)

model_path = os.path.expandvars(
    config["model_path"]
)

evals_path = os.path.expandvars(
    config["evals_path"]
)

norm_path = os.path.expandvars(
    config["norm_params_path"]
)

datasets = list(
    config["target_manipulations"].keys()
)

all_metric_rows = []

start_total = time.time()

all_metric_rows = []

# Evaluate for all combinations of datasets/backbones/manipualations/features/models
for dataset in datasets:

    manipulation_groups = config["target_manipulations"][dataset]

    for backbone in config["extractor_name"]:

        feature_cfgs = config["feature_keys"][backbone]

        backbone_model_root = os.path.join(model_path, backbone)

        for group_name, group_cfg in manipulation_groups.items():

            feature_subdir = resolve_feature_subdir(group_name)

            allowed_models = group_cfg["models"]
            manip_list = group_cfg["manipulations"]
            apply_flag = group_cfg.get("apply_transform", False)


            for feature_key, feat_cfg in feature_cfgs.items():

                input_dim = feat_cfg["input_dim"]
                output_dim = feat_cfg["output_dim"]
                spatial_size = feat_cfg["spatial_size"]

                if backbone == "swinv2":
                    normalize_flag = False
                else:
                    normalize_flag = not feature_key.endswith("feat3")

                feature_dir = os.path.join(
                    dataset_path,
                    dataset,
                    f"{backbone}_features",
                    "augmented_test",
                    feature_subdir
                )

                meta_path = os.path.join(
                    feature_dir,
                    "meta.jsonl"
                )

                meta_rows = []

                with open(meta_path) as f:
                    for line in f:
                        meta_rows.append(
                            json.loads(line)
                        )

                index = defaultdict(dict)

                for m in meta_rows:

                    oid = m["original_id"]
                    manip = m["manipulation"]
                    row = m["row_idx"]

                    index[oid][manip] = row

                pairs_by_manip = defaultdict(list)

                for oid, entry in index.items():

                    if "resized" not in entry:
                        continue

                    o_idx = entry["resized"]

                    for manip, m_idx in entry.items():

                        if manip == "resized":
                            continue

                        pairs_by_manip[manip].append(
                            (o_idx, m_idx, manip)
                        )

                # Limit per manip, as this proved sufficient
                MAX_PER_MANIP = 200

                sampled_pairs_by_manip = {}

                for manip, items in pairs_by_manip.items():

                    items = list(set(items))

                    k = min(MAX_PER_MANIP, len(items))

                    sampled_pairs_by_manip[manip] = random.sample(
                        items,
                        k
                    )

                for model_type in allowed_models:

                    model_params = config["models"].get(
                        model_type,
                        {}
                    )

                    for manip in manip_list:

                        manip_pairs = sampled_pairs_by_manip.get(
                            manip,
                            []
                        )

                        if len(manip_pairs) == 0:
                            continue

                        dataset_obj = PairedFeatureDataset(
                            path_prefix=feature_dir,
                            feature_key=feature_key,
                            pairs=np.asarray(
                                manip_pairs,
                                dtype=object
                            ),
                            manipulation=None,
                        )

                        if normalize_flag:

                            feature_dataset = DatasetNormalize(
                                dataset_obj,
                                dataset,
                                "resized",
                                feature_key,
                                os.path.join(norm_params_path,
                                             "convnext"),
                                False
                            )


                        loader = DataLoader(
                            dataset_obj,
                            batch_size=128,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=test_collate_fn,
                            pin_memory=True,
                            persistent_workers=True
                        )

                        model = load_model(
                            dataset,
                            backbone_model_root,
                            normalize_flag,
                            feature_subdir,
                            manip,
                            model_type,
                            input_dim,
                            output_dim,
                            model_params,
                            device,
                            feature_key,
                            spatial_size ** 2,
                            apply_transform=apply_flag,
                        )

                        if model is None:
                            continue

                        rows = compute_metrics(
                            model,
                            loader,
                            spatial_size,
                            input_dim,
                            dataset,
                            backbone,
                            feature_key,
                            model_type,
                            group_name
                        )

                        all_metric_rows.extend(rows)

        df = pd.DataFrame(all_metric_rows)

        output_dir = os.path.join(
            evals_path,
            "bias_group_analysis"
        )

        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(
            os.path.join(
                output_dir,
                f"wxb_metrics_{backbone}.csv"
            ),
            index=False
        )
