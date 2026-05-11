import os
import glob
import json
import yaml
import re
import random
import numpy as np
import timm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from prepare_datasets.feature_dataset_impl import (
    PairedFeatureDataset,
    DatasetNormalize,
    test_collate_fn
)

from model_implementations import (
    resolve_feature_subdir,
    load_model
)
from utils import set_seed
from torch.utils.data import DataLoader


set_seed(42)

"""
Evaluation pipeline for evaluating semantic quality of mapped features.

This script:
- Loads pretrained ConvNeXt classifiers and intermediate feature heads
- Evaluates classification performance on:
  (1) original features
  (2) manipulated/edit features
  (3) mapped features (via learned feature-to-feature models)

"""

class ConvNeXtTail(nn.Module):
    def __init__(self, model, start_stage):
        super().__init__()

        self.tail_stages = model.stages[start_stage + 1:]
        self.norm_pre = model.norm_pre
        self.head = model.head

    def forward(self, x):

        for stage in self.tail_stages:
            x = stage(x)

        x = self.norm_pre(x)

        return self.head(x)


def get_classifier_head(
    model,
    feature_key,
    device
):

    stage_map = {
        "feat0": 0,
        "feat1": 1,
        "feat2": 2,
        "feat3": 3
    }

    start_stage = stage_map[feature_key]

    head = ConvNeXtTail(
        model,
        start_stage
    ).to(device)

    head.eval()

    return head


def evaluate_loader(
    loader,
    classifier,
    device,
    save_path,
    mapping_model=None,
    dataset=None,
    normalized=False,
    use_target=False,
):
    """
    Runs a classifier over feature embeddings and stores softmax outputs.

    Supports:
    - original features
    - target (manipulated) features
    - mapped features (via learned transformation model)

    Optionally applies normalization inversion before classification.
    """

    probs = []
    targets = []

    with torch.inference_mode():
        for batch in loader:
            (
                orig_feats,
                target_feats,
                orig_labels,
                target_labels,
                manips,
                orig_indices,
                target_indices
            ) = batch
            feats = target_feats if use_target else orig_feats
            feats = feats.to(device)

            if mapping_model is not None:
                feats = mapping_model(feats)
                if normalized and dataset is not None:
                    feats = dataset.denormalize(feats)

            logits = classifier(feats)
            probs.append(F.softmax(logits, dim=1).cpu())

            if use_target:
                targets.append(target_labels.cpu())
            else:
                targets.append(orig_labels.cpu())

    probs = torch.cat(probs)
    targets = torch.cat(targets)

    torch.save(
        {
            "probs": probs,
            "targets": targets
        },
        save_path
    )


def main():
    start_time = time.time()

    with open("../config/test_classifier.yaml") as f:
        config = yaml.safe_load(f)

    model_path = os.path.expandvars(config["model_path"])
    dataset_path = os.path.expandvars(config["dataset_path"])
    evals_path = os.path.expandvars(config["evals_path"])
    norm_params_path = os.path.expandvars(config["norm_params_path"])

    os.makedirs(evals_path, exist_ok=True)

    batch_size = config["batch_size"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for ft_model_name in config["ft_models"]:
        print(f"\nUsing classifier: {ft_model_name}")

        save_dir = os.path.join(
            evals_path,
            f"convnext/class_probs/{'augmented_' if 'augmented' in ft_model_name else ''}head/"
        )

        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(
            model_path,
            ft_model_name
        )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)

        classifier_model = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=196
        )

        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu"
        )

        state_dict = {
            k.replace("model.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }

        classifier_model.load_state_dict(
            state_dict,
            strict=False
        )

        classifier_model.to(device)
        classifier_model.eval()

        # build heads
        classifier_heads = {}

        for feature_key in config["feature_keys"]:

            classifier_heads[feature_key] = get_classifier_head(
                classifier_model,
                feature_key,
                device
            ).eval()

        for dataset in config["target_manipulations"]:
            groups = config["target_manipulations"][dataset]

            for group_name, group_cfg in groups.items():

                allowed_models = group_cfg["models"]
                manip_list = group_cfg["manipulations"]
                apply_flag = group_cfg.get("apply_transform", False)

                feature_subdir = resolve_feature_subdir(
                    group_name
                )

                root_dir = os.path.join(
                    dataset_path,
                    dataset
                )

                test_feature_dir = os.path.join(
                    root_dir,
                    f"convnext_features",
                    "augmented_test",
                    feature_subdir
                )
                meta_path = os.path.join(
                    test_feature_dir,
                    "meta.jsonl"
                )

                row_to_meta = {}

                with open(meta_path) as f:
                    for line in f:
                        m = json.loads(line)
                        row_to_meta[m["row_idx"]] = m

                pairs_path = os.path.join(
                    test_feature_dir,
                    "pairs.npy"
                )

                pairs_all = np.load(
                    pairs_path,
                    allow_pickle=True
                )

                for target_manipulation in manip_list:
                    pairs = pairs_all[
                        pairs_all[:, 2] == target_manipulation
                    ]
                    print(
                        f"{dataset} | {target_manipulation} | "
                        f"pairs: {len(pairs)}"
                    )

                    for feature_key, feat_cfg in config["feature_keys"].items():
                        print(f"Evaluating {feature_key}")

                        input_dim = feat_cfg["input_dim"]
                        output_dim = feat_cfg["output_dim"]
                        spatial_size = feat_cfg["spatial_size"]

                        num_feature_vectors = spatial_size ** 2

                        classifier = classifier_heads[
                            feature_key
                        ]

                        feature_dataset = PairedFeatureDataset(
                            path_prefix=test_feature_dir,
                            feature_key=feature_key,
                            pairs=pairs,
                            manipulation=target_manipulation
                        )
                        normalized = feat_cfg.get("normalized", True)

                        if normalized:

                            feature_dataset = DatasetNormalize(
                                feature_dataset,
                                dataset,
                                "resized",
                                feature_key,
                                os.path.join(norm_params_path,
                                             "convnext"),
                                False
                            )

                        loader = DataLoader(
                            feature_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=test_collate_fn
                        )

                        save_path = os.path.join(
                            save_dir,
                            f"probs_orig_{dataset}_{target_manipulation}_{feature_key}.pt"
                        )

                        evaluate_loader(
                            loader,
                            classifier,
                            device,
                            save_path,
                        )

                        edit_save_path = os.path.join(
                            save_dir,
                            f"probs_edit_{dataset}_{target_manipulation}_{feature_key}.pt"
                        )

                        evaluate_loader(
                            loader,
                            classifier,
                            device,
                            edit_save_path,
                            use_target=True,
                            mapping_model=None,
                            dataset=feature_dataset,
                            normalized=normalized
                        )

                        for model_type in allowed_models:

                            model_params = config["models"].get(
                                model_type,
                                {}
                            )
                            extractor_model_path = os.path.join(
                                model_path,
                                "convnext"
                            )
                            model = load_model(
                                dataset,
                                extractor_model_path,
                                normalized,
                                feature_subdir,
                                target_manipulation,
                                model_type,
                                input_dim,
                                output_dim,
                                model_params or {},
                                device,
                                feature_key,
                                num_feature_vectors=num_feature_vectors,
                                apply_transform=apply_flag,
                            )

                            if model is None:
                                continue

                            mode = f"{model_type}_{target_manipulation}" if apply_flag else model_type

                            mapped_path = os.path.join(
                                save_dir,
                                f"probs_mapped_{dataset}_{target_manipulation}_{feature_key}_{mode}.pt"
                            )

                            evaluate_loader(
                                loader,
                                classifier,
                                device,
                                mapped_path,
                                mapping_model=model,
                                dataset=feature_dataset,
                                normalized=normalized
                            )

    print(
        "Total time:",
        time.time() - start_time
    )


if __name__ == "__main__":
    main()
