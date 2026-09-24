import os
import json
import yaml
import time
from PIL import Image

import numpy as np
import torch
import timm

from torch.utils.data import DataLoader

from prepare_datasets.feature_dataset_impl import (
    PairedFeatureDataset,
    DatasetNormalize,
    test_collate_fn,
)

from FeatInv.featinv_reconstructor_conv import InputReconstructorConv
from FeatInv.featinv_reconstructor_swinv2 import InputReconstructorSwinV2
from FeatInv.featinv_reconstructor_dinov3 import InputReconstructorDinoV3

from dinov3_extractor import (
    DINOv3FeatureExtractor,
    DEFAULT_MODEL_NAME,
)

from eval_functions import Evaluator
from eval_helpers import process_features, process_image
from model_implementations import load_model, apply_spatial_transform
from utils import (
    get_img_path_by_id,
    set_seed,
    resolve_feature_subdir,
    get_shared_ids,
    filter_pairs_by_ids,
    sample_ids,
)

set_seed(42)

from timm.models.swin_transformer_v2 import SwinTransformerV2

class SwinV2StagedFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,  # full model
        )
        # Ensure we can access stages
        assert isinstance(self.backbone, SwinTransformerV2)

    def forward_features(self, x: torch.Tensor, stage: int):
        """
        Return features from the given SwinV2 stage (0–3) as [B, C, H, W].
        """
        x = self.backbone.patch_embed(x)

        # stages are in self.backbone.layers, indexed 0..3
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)

            if i == stage:
                # Normalize layout to [B, C, H, W]
                if x.ndim == 3:
                    # [B, HW, C] -> [B, C, H, W]
                    B, HW, C = x.shape
                    H = W = int(HW ** 0.5)
                    x = x.permute(0, 2, 1).view(B, C, H, W)
                elif x.ndim == 4:
                    # Could be [B, C, H, W] or [B, H, W, C]
                    B, C, H, W = x.shape
                    # Heuristic: if last dim looks like channels, permute
                    if C < x.shape[-1]:
                        # Assume [B, H, W, C]
                        x = x.permute(0, 3, 1, 2)
                    # Now treat as [B, C, H, W]
                else:
                    raise RuntimeError(
                        f"Unexpected feature tensor ndim={x.ndim}, shape={tuple(x.shape)}"
                    )

                return x.contiguous()

        # If we exit the loop (e.g. stage >= num_layers), treat final x similarly
        if x.ndim == 3:
            B, HW, C = x.shape
            H = W = int(HW ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        elif x.ndim == 4:
            B, C, H, W = x.shape
            if C < x.shape[-1]:
                x = x.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(
                f"Unexpected feature tensor ndim={x.ndim}, shape={tuple(x.shape)}"
            )

        return x.contiguous()

    def forward(self, x):
        return self.backbone(x)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extractor",
        type=str,
        default=None,
        choices=("convnext", "swinv2", "dinov3"),
        help=(
            "Extractor to evaluate. If omitted, all configured extractors are used."
        ),
    )

    parser.add_argument(
        "--feature-key",
        type=str,
        default=None,
        help=(
            "Feature key to evaluate (e.g. feat3). "
            "If omitted, all valid feature keys are used."
        ),
    )

    parser.add_argument(
        "--model",
        dest="model_type",
        type=str,
        default=None,
        # Do NOT restrict choices here if you want orthogonal_procrustes/ridge too
        help=(
            "Mapping model to evaluate. If omitted, all models allowed "
            "by the target group are used."
        ),
    )

    parser.add_argument(
        "--target-group",
        type=str,
        default=None,
        help=(
            "Target manipulation group (e.g. direct). "
            "If omitted, all groups are used."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    return parser.parse_args()

from itertools import product


def generate_eval_combinations(config, extractor_filter=None, feature_key_filter=None,
                               model_filter=None, target_group_filter=None):
    """
    Generate (extractor, feature_key, model_type, target_group) tuples
    consistent with the config and optional filters.
    """
    combinations = []

    # Extractors
    configured_extractors = config.get("extractor_name", ["convnext", "swinv2", "dinov3"])
    extractors = (
        [extractor_filter] if extractor_filter is not None
        else configured_extractors
    )
    extractors = [e for e in extractors if e in configured_extractors]

    # Losses are not part of the combination here; they’re handled inside main().

    # Target groups
    manipulations_by_dataset = config["target_manipulations"]
    target_groups = []
    for dataset, groups in manipulations_by_dataset.items():
        for group_name, group_cfg in groups.items():
            if target_group_filter is not None and group_name != target_group_filter:
                continue
            target_groups.append((dataset, group_name, group_cfg))

    for extractor in extractors:
        # Feature keys for this extractor
        extractor_feature_cfgs = config["feature_keys"].get(extractor, {})
        feature_keys = (
            [feature_key_filter] if feature_key_filter is not None
            else list(extractor_feature_cfgs.keys())
        )
        feature_keys = [fk for fk in feature_keys if fk in extractor_feature_cfgs]

        for dataset, group_name, group_cfg in target_groups:
            allowed_models = group_cfg["models"]
            if model_filter is not None:
                if model_filter not in allowed_models:
                    # Skip: this model not allowed for this group
                    continue
                models_to_run = [model_filter]
            else:
                models_to_run = list(allowed_models)

            for feature_key in feature_keys:
                for model_type in models_to_run:
                    combinations.append(
                        (extractor, feature_key, model_type, group_name)
                    )

    return combinations

OLD_WORK_DIR = "/dss/work/kedi1373"
NEW_WORK_DIR = "/dss/work/zual3488"

def fix_path(p: str) -> str:
    if p.startswith(OLD_WORK_DIR):
        return NEW_WORK_DIR + p[len(OLD_WORK_DIR):]
    return p

def load_pairs_and_meta(extractor_name, dataset, dataset_path, feature_subdir, manipulation):
    base_root = os.path.join(dataset_path, dataset)

    feature_dir = os.path.join(
        base_root,
        f"{extractor_name}_features",
        "augmented_test",
        feature_subdir,
    )

    meta_path = os.path.join(feature_dir, "meta.jsonl")

    row_to_meta = {}
    with open(meta_path) as f:
        for line in f:
            m = json.loads(line)
            # Fix image path if it still points to kedi1373
            m["path"] = fix_path(m["path"])
            row_to_meta[m["row_idx"]] = m

    pairs = np.load(os.path.join(feature_dir, "pairs.npy"), allow_pickle=True)
    pairs = pairs[pairs[:, 2] == manipulation]

    return pairs, row_to_meta, feature_dir


def get_backbone(name, device):
    if name == "convnext":
        return timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            features_only=True,
        ).eval().to(device)

    if name == "swinv2":
        return SwinV2StagedFeatureExtractor(
            "swinv2_base_window12to24_192to384_22kft1k",
            pretrained=True,
        ).eval().to(device)

    if name == "dinov3":
        backbone = DINOv3FeatureExtractor(
            model_name=DEFAULT_MODEL_NAME,
            hf_token=os.environ.get("HF_TOKEN"),
            device=device,
        )

        return backbone

    raise ValueError(f"Unknown backbone name: {name}")


def run_eval_loop(test_loader, step_fn):
    with torch.inference_mode():
        for batch in test_loader:
            step_fn(batch)


def run_spatial_baseline(
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
    normalizer,
    target_recon_paths,
    loss_name,
):

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
            if normalized and normalizer is not None:
                feat = normalizer.denormalize(feat)

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
                loss_name=loss_name,
            )

    run_eval_loop(test_loader, step)

def run_mapping_models(
    *,
    allowed_models,
    config,
    dataset,
    extractor,
    extractor_model_path,
    normalized,
    normalizer,
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
    apply_flag,
    loss_name,
):

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
            loss_name=loss_name,
            num_feature_vectors=num_feature_vectors,
            apply_transform=apply_flag,
        )

        if model is None:
            print(
                "DEBUG: load_model returned None for",
                dataset, extractor, feature_key, target_manipulation, model_type,
            )
            continue

        def step(batch):
            orig_feats, _, _, _, _, orig_indices, _ = batch
            orig_feats = orig_feats.to(device)
            mapped = model(orig_feats)

            for i, idx in enumerate(orig_indices):
                meta = row_to_meta[idx]
                orig_id = meta["original_id"]

                feat = mapped[i]
                if normalized and normalizer is not None:
                    feat = normalizer.denormalize(feat)

                img = process_features(feat.cpu().numpy(), reconstructor)

                mode = (
                    f"{model_type}_{target_manipulation}_loss_{loss_name}"
                    if apply_flag
                    else f"{model_type}_loss_{loss_name}"
                )

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
                    target_reconstructed_img_path=target_recon_paths.get(orig_id),
                    eval_save_dir=save_dir,
                    feature_key=feature_key,
                    showFig=False,
                    loss_name=loss_name,
                )

        run_eval_loop(test_loader, step)

def get_selected_dataset_and_group(
    config,
    target_group,
):
    """
    Resolve the dataset and group configuration.

    The current configuration contains one dataset. This function still
    searches explicitly so that the code fails clearly if the group is
    missing or becomes ambiguous later.
    """

    matches = []

    for dataset, groups in config["target_manipulations"].items():
        if target_group in groups:
            matches.append(
                (
                    dataset,
                    groups[target_group],
                )
            )

    if not matches:
        raise ValueError(
            f"Target group {target_group!r} was not found in "
            "config['target_manipulations']."
        )

    if len(matches) > 1:
        datasets = [dataset for dataset, _ in matches]

        raise ValueError(
            f"Target group {target_group!r} occurs in multiple datasets: "
            f"{datasets}. Add a --dataset argument to disambiguate."
        )

    return matches[0]


def stable_seed(*values):
    """
    Generate a reproducible 32-bit seed from string-like values.
    """

    import hashlib

    text = "::".join(str(value) for value in values)

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def select_ids_deterministically(
    row_to_meta,
    dataset,
    target_group,
    subset_size,
):
    """
    Select a reproducible subset of original IDs.

    The same dataset/group combination receives the same subset in every
    array task, independent of Python hash randomization or metadata order.
    """

    all_ids = sorted({
        meta["original_id"]
        for meta in row_to_meta.values()
    })

    if not all_ids:
        return set()

    if subset_size is None:
        return set(all_ids)

    if subset_size >= len(all_ids):
        return set(all_ids)

    rng = np.random.default_rng(
        stable_seed(
            dataset,
            target_group,
        )
    )

    selected = rng.choice(
        all_ids,
        size=subset_size,
        replace=False,
    )

    return set(selected.tolist())


def initialize_reconstructor(
    extractor,
    feature_key,
    device,
):
    """
    Initialize the backbone and feature reconstructor once per Slurm job.
    """

    backbone = get_backbone(
        extractor,
        device,
    )

    if extractor == "convnext":
        reconstructor = InputReconstructorConv(
            backbone,
            feature_key,
        )

    elif extractor == "swinv2":
        reconstructor = InputReconstructorSwinV2(
            backbone,
            feature_key,
        )

    elif extractor == "dinov3":
        reconstructor = InputReconstructorDinoV3(
            feature_model=backbone,
            feature_key=feature_key,
        )

    else:
        raise ValueError(
            f"Unknown extractor: {extractor}"
        )

    return backbone, reconstructor


def find_target_path(
    row_to_meta,
    original_id,
    manipulation,
):
    """
    Find the target image associated with one original image ID.
    """

    for meta in row_to_meta.values():
        if (
            meta["original_id"] == original_id
            and meta["manipulation"] == manipulation
        ):
            return meta["path"]

    return None


def find_target_pair(
    pairs,
    row_to_meta,
    original_id,
):
    """
    Find the feature pair whose source image has original_id.
    """

    for pair in pairs:
        source_idx = pair[0]

        if row_to_meta[source_idx]["original_id"] == original_id:
            return pair

    return None


def save_reconstructed_image(
    image,
    save_path,
):
    """
    Save PIL, NumPy, or list/tuple-wrapped reconstruction outputs.
    """

    if isinstance(image, (list, tuple)):
        image = image[0]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Unsupported reconstructed image type: "
            f"{type(image)}"
        )

    image.save(save_path)


def precompute_target_reconstructions(
    *,
    selected_ids,
    pairs,
    row_to_meta,
    manipulation,
    extractor,
    test_dataset,
    reconstructor,
    normalized,
    normalizer,
    extractor_eval_path,
    feature_subdir,
    dataset,
    feature_key,
):
    """
    Reconstruct target images once for one manipulation.
    """

    target_recon_paths = {}

    target_eval_dir = os.path.join(
        extractor_eval_path,
        feature_subdir,
        dataset,
        feature_key,
        manipulation,
        "targets",
    )

    os.makedirs(
        target_eval_dir,
        exist_ok=True,
    )

    for original_id in sorted(selected_ids):
        try:
            target_img_path = find_target_path(
                row_to_meta=row_to_meta,
                original_id=original_id,
                manipulation=manipulation,
            )

            if target_img_path is None:
                print(
                    "SKIP: no target image found for "
                    f"original_id={original_id}, "
                    f"manipulation={manipulation}",
                    flush=True,
                )
                continue

            if extractor == "dinov3":
                target_pair = find_target_pair(
                    pairs=pairs,
                    row_to_meta=row_to_meta,
                    original_id=original_id,
                )

                if target_pair is None:
                    print(
                        "SKIP: no target pair found for "
                        f"original_id={original_id}",
                        flush=True,
                    )
                    continue

                target_idx = target_pair[1]

                target_feat = (
                    test_dataset.backend.get_feat(target_idx)
                )

                if normalized and normalizer is not None:
                    target_feat = normalizer.denormalize(
                        target_feat
                    )

                target_img = reconstructor.reconstruct(
                    target_feat
                )

            else:
                source_image = Image.open(
                    target_img_path
                ).convert("RGB")

                target_img = process_image(
                    source_image,
                    reconstructor,
                    extractor,
                )

            target_save_path = os.path.join(
                target_eval_dir,
                f"{original_id}_target_recon.png",
            )

            save_reconstructed_image(
                target_img,
                target_save_path,
            )

            target_recon_paths[original_id] = target_save_path

        except Exception as exc:
            print(
                "SKIP: target reconstruction failed for "
                f"extractor={extractor}, "
                f"feature_key={feature_key}, "
                f"manipulation={manipulation}, "
                f"original_id={original_id}: {exc}",
                flush=True,
            )

    return target_recon_paths
def main(
    config,
    extractor_name,
    feature_key,
    model_type,
    target_group,
    run_name=None,
):
    start = time.time()

    print("=" * 80)
    print("Starting mapping evaluation")
    print(f"Extractor:      {extractor_name}")
    print(f"Feature key:    {feature_key}")
    print(f"Mapping model:  {model_type}")
    print(f"Target group:   {target_group}")
    print(f"Run name:       {run_name}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Resolve paths and device.
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = os.path.expandvars(
        config["model_path"]
    )

    norm_path = os.path.expandvars(
        config["norm_params_path"]
    )

    dataset_path = os.path.expandvars(
        config["dataset_path"]
    )

    eval_path = os.path.expandvars(
        config["evals_path"]
    )

    batch_size = config["batch_size"]
    subset_size = config.get("subset_size")

    # ------------------------------------------------------------------
    # Validate the selected extractor.
    # ------------------------------------------------------------------

    configured_extractors = config.get(
        "extractor_name",
        ["convnext", "swinv2", "dinov3"],
    )

    if extractor_name not in configured_extractors:
        raise ValueError(
            f"Extractor {extractor_name!r} is not configured. "
            f"Available extractors: {configured_extractors}"
        )

    # ------------------------------------------------------------------
    # Resolve dataset and target group.
    # ------------------------------------------------------------------

    dataset, group_cfg = get_selected_dataset_and_group(
        config=config,
        target_group=target_group,
    )

    feature_subdir = resolve_feature_subdir(
        target_group
    )

    apply_flag = group_cfg.get(
        "apply_transform",
        False,
    )

    allowed_models = group_cfg["models"]

    if model_type not in allowed_models:
        raise ValueError(
            f"Model {model_type!r} is not allowed for target group "
            f"{target_group!r}. Available models: {allowed_models}"
        )

    manipulations = list(
        group_cfg["manipulations"]
    )

    if not manipulations:
        raise RuntimeError(
            f"Target group {target_group!r} contains no manipulations."
        )

    # ------------------------------------------------------------------
    # Resolve feature configuration.
    # ------------------------------------------------------------------

    extractor_feature_cfgs = config["feature_keys"].get(
        extractor_name
    )

    if extractor_feature_cfgs is None:
        raise KeyError(
            f"No feature configuration exists for extractor "
            f"{extractor_name!r}."
        )

    if feature_key not in extractor_feature_cfgs:
        raise ValueError(
            f"Feature key {feature_key!r} is not configured for "
            f"extractor {extractor_name!r}. Available keys: "
            f"{list(extractor_feature_cfgs.keys())}"
        )

    feat_cfg = extractor_feature_cfgs[feature_key]

    input_dim = feat_cfg["input_dim"]
    output_dim = feat_cfg["output_dim"]
    spatial_size = feat_cfg["spatial_size"]
    normalized = feat_cfg["normalize_features"]
    num_feature_vectors = spatial_size * spatial_size

    loss_functions = config.get(
        "loss_functions",
        ["mse"],
    )

    print(f"Dataset:        {dataset}")
    print(f"Feature config: {feat_cfg}")
    print(f"Manipulations:  {manipulations}")
    print(f"Loss functions: {loss_functions}")
    print(f"Normalized:     {normalized}")
    print(f"Apply transform: {apply_flag}")
    print(f"Device:         {device}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load the backbone and reconstructor once.
    #
    # The same reconstructor is reused for all manipulations in this
    # target group.
    # ------------------------------------------------------------------

    try:
        backbone, reconstructor = initialize_reconstructor(
            extractor=extractor_name,
            feature_key=feature_key,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize backbone/reconstructor for "
            f"extractor={extractor_name}, "
            f"feature_key={feature_key}: {exc}"
        ) from exc

    # Keep the backbone alive because some reconstructors access it
    # indirectly.
    _ = backbone

    # ------------------------------------------------------------------
    # Prepare output and evaluator objects.
    # ------------------------------------------------------------------

    evaluator = Evaluator(
        eval_path,
        extractor_name,
    )

    extractor_eval_path = os.path.join(
        eval_path,
        extractor_name,
    )

    os.makedirs(
        extractor_eval_path,
        exist_ok=True,
    )

    extractor_model_path = os.path.join(
        model_path,
        extractor_name,
    )

    os.makedirs(
        extractor_model_path,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Use fewer workers than requested CPUs to avoid oversubscription.
    # ------------------------------------------------------------------

    slurm_cpus = int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            "4",
        )
    )

    num_workers = max(
        0,
        min(4, slurm_cpus - 1),
    )

    print(
        f"DataLoader workers: {num_workers}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Process all manipulations in the selected target group.
    # ------------------------------------------------------------------

    for manipulation_index, manipulation in enumerate(
        manipulations,
        start=1,
    ):
        print("\n" + "=" * 80)
        print(
            f"Manipulation {manipulation_index}/{len(manipulations)}: "
            f"{manipulation}"
        )
        print("=" * 80)

        # --------------------------------------------------------------
        # Load pairs and metadata for this manipulation.
        # --------------------------------------------------------------

        try:
            pairs, row_to_meta, feature_root = (
                load_pairs_and_meta(
                    extractor_name=extractor_name,
                    dataset=dataset,
                    dataset_path=dataset_path,
                    feature_subdir=feature_subdir,
                    manipulation=manipulation,
                )
            )

        except Exception as exc:
            print(
                "SKIP: cannot load pairs/meta for "
                f"dataset={dataset}, "
                f"group={target_group}, "
                f"manipulation={manipulation}, "
                f"extractor={extractor_name}: {exc}",
                flush=True,
            )
            continue

        if len(pairs) == 0:
            print(
                "SKIP: no pairs found for "
                f"manipulation={manipulation}",
                flush=True,
            )
            continue

        # --------------------------------------------------------------
        # Select a deterministic subset of IDs.
        #
        # This is done independently of metadata insertion order and
        # remains identical across Slurm jobs for this dataset/group.
        # --------------------------------------------------------------

        selected_ids = select_ids_deterministically(
            row_to_meta=row_to_meta,
            dataset=dataset,
            target_group=target_group,
            subset_size=subset_size,
        )

        pairs = filter_pairs_by_ids(
            pairs,
            row_to_meta,
            selected_ids,
        )

        if len(pairs) == 0:
            print(
                "SKIP: no pairs remain after selected-ID filtering "
                f"for manipulation={manipulation}",
                flush=True,
            )
            continue

        print(
            f"Selected IDs: {len(selected_ids)}",
            flush=True,
        )

        print(
            f"Filtered pairs: {len(pairs)}",
            flush=True,
        )

        # --------------------------------------------------------------
        # Build dataset and optional normalizer.
        # --------------------------------------------------------------

        try:
            test_dataset = PairedFeatureDataset(
                path_prefix=feature_root,
                feature_key=feature_key,
                pairs=pairs,
                manipulation=manipulation,
            )
        except Exception as exc:
            print(
                "SKIP: cannot create PairedFeatureDataset for "
                f"manipulation={manipulation}: {exc}",
                flush=True,
            )
            continue

        normalizer = None

        if normalized:
            extractor_norm_path = os.path.join(
                norm_path,
                extractor_name,
            )

            expected_norm_file = os.path.join(
                extractor_norm_path,
                dataset,
                f"resized_{feature_key}.pt",
            )

            print(
                "Expected normalization file:",
                expected_norm_file,
                flush=True,
            )

            if not os.path.isfile(expected_norm_file):
                print(
                    "SKIP: normalization file does not exist: "
                    f"{expected_norm_file}",
                    flush=True,
                )
                continue

            try:
                normalizer = DatasetNormalize(
                    dataset=test_dataset,
                    source_dataset=dataset,
                    manip="resized",
                    feat_key=feature_key,
                    norm_params_path=extractor_norm_path,
                    recalc_norm_params=False,
                )

                print(
                    "Loaded normalizer:",
                    f"mean={tuple(normalizer.mean.shape)}",
                    f"std={tuple(normalizer.std.shape)}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    "SKIP: cannot load normalizer for "
                    f"extractor={extractor_name}, "
                    f"dataset={dataset}, "
                    f"feature_key={feature_key}: {exc}",
                    flush=True,
                )
                continue

        # --------------------------------------------------------------
        # Reconstruct target images once for this manipulation.
        # --------------------------------------------------------------

        target_recon_paths = (
            precompute_target_reconstructions(
                selected_ids=selected_ids,
                pairs=pairs,
                row_to_meta=row_to_meta,
                manipulation=manipulation,
                extractor=extractor_name,
                test_dataset=test_dataset,
                reconstructor=reconstructor,
                normalized=normalized,
                normalizer=normalizer,
                extractor_eval_path=extractor_eval_path,
                feature_subdir=feature_subdir,
                dataset=dataset,
                feature_key=feature_key,
            )
        )

        if not target_recon_paths:
            print(
                "SKIP: no target reconstructions were created for "
                f"manipulation={manipulation}",
                flush=True,
            )
            continue

        print(
            f"Target reconstructions: "
            f"{len(target_recon_paths)}",
            flush=True,
        )

        # --------------------------------------------------------------
        # Create DataLoader.
        # --------------------------------------------------------------

        try:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=torch.cuda.is_available(),
                collate_fn=lambda batch: test_collate_fn(
                    batch,
                    normalizer=normalizer,
                ),
            )

        except Exception as exc:
            print(
                "SKIP: cannot create DataLoader for "
                f"extractor={extractor_name}, "
                f"dataset={dataset}, "
                f"feature_key={feature_key}, "
                f"manipulation={manipulation}: {exc}",
                flush=True,
            )
            continue

        # --------------------------------------------------------------
        # Run the spatial baseline for geometric transformations.
        # --------------------------------------------------------------

        if apply_flag:
            try:
                run_spatial_baseline(
                    test_loader,
                    manipulation,
                    extractor_eval_path,
                    feature_subdir,
                    dataset,
                    feature_key,
                    row_to_meta,
                    evaluator,
                    reconstructor,
                    test_dataset,
                    normalized,
                    normalizer,
                    target_recon_paths,
                    "mse_median_cos",
                )

            except Exception as exc:
                print(
                    "SKIP: spatial baseline failed for "
                    f"dataset={dataset}, "
                    f"extractor={extractor_name}, "
                    f"feature_key={feature_key}, "
                    f"manipulation={manipulation}: {exc}",
                    flush=True,
                )

        # --------------------------------------------------------------
        # Evaluate only the model assigned by the Slurm array task.
        # --------------------------------------------------------------

        for loss_name in loss_functions:
            print(
                f"Evaluating model={model_type}, "
                f"loss={loss_name}",
                flush=True,
            )

            try:
                run_mapping_models(
                    allowed_models=[model_type],
                    config=config,
                    dataset=dataset,
                    extractor=extractor_name,
                    extractor_model_path=extractor_model_path,
                    normalized=normalized,
                    normalizer=normalizer,
                    feature_subdir=feature_subdir,
                    target_manipulation=manipulation,
                    device=device,
                    feature_key=feature_key,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_feature_vectors=num_feature_vectors,
                    test_loader=test_loader,
                    row_to_meta=row_to_meta,
                    evaluator=evaluator,
                    reconstructor=reconstructor,
                    test_dataset=test_dataset,
                    target_recon_paths=target_recon_paths,
                    extractor_eval_path=extractor_eval_path,
                    apply_flag=apply_flag,
                    loss_name=loss_name,
                )

            except Exception as exc:
                print(
                    "SKIP: mapping-model evaluation failed for "
                    f"dataset={dataset}, "
                    f"group={target_group}, "
                    f"extractor={extractor_name}, "
                    f"feature_key={feature_key}, "
                    f"model={model_type}, "
                    f"manipulation={manipulation}, "
                    f"loss={loss_name}: {exc}",
                    flush=True,
                )

    print(
        f"Total evaluation time: {time.time() - start:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    args = parse_args()

    with open("../config/test_mapping_cars_baseline.yaml") as f:
        config = yaml.safe_load(f)

    start_time = time.time()

    # If all key args are provided, run a single combination
    if all([
        args.extractor is not None,
        args.feature_key is not None,
        args.model_type is not None,
        args.target_group is not None,
    ]):
        combos = [(
            args.extractor,
            args.feature_key,
            args.model_type,
            args.target_group,
        )]
    else:
        # Generate all valid combinations from the config
        combos = generate_eval_combinations(
            config,
            extractor_filter=args.extractor,
            feature_key_filter=args.feature_key,
            model_filter=args.model_type,
            target_group_filter=args.target_group,
        )

        print(
            f"Running evaluation over {len(combos)} combinations "
            f"(extractor={args.extractor}, feature_key={args.feature_key}, "
            f"model={args.model_type}, target_group={args.target_group})",
            flush=True,
        )

    for extractor_name, feature_key, model_type, target_group in combos:
        print(
            "\n" + "=" * 80,
            flush=True,
        )
        print(
            f"EVAL COMBINATION: extractor={extractor_name}, "
            f"feature_key={feature_key}, model={model_type}, "
            f"target_group={target_group}",
            flush=True,
        )
        print("=" * 80)

        try:
            main(
                config=config,
                extractor_name=extractor_name,
                feature_key=feature_key,
                model_type=model_type,
                target_group=target_group,
                run_name=args.run_name,
            )
        except Exception as e:
            print(
                f"EVAL FAILED for extractor={extractor_name}, "
                f"feature_key={feature_key}, model={model_type}, "
                f"target_group={target_group}: {e}",
                flush=True,
            )
            # Continue with next combination instead of aborting everything
            continue

    print(
        f"Total evaluation time: {time.time() - start_time:.2f}s",
        flush=True,
    )