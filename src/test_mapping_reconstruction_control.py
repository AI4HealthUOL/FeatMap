import csv
import gc
import json
import os
import time
from contextlib import nullcontext

import numpy as np
import timm
import torch
import yaml

from torch.utils.data import DataLoader
from timm.models.swin_transformer_v2 import SwinTransformerV2

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
from eval_helpers import process_features
from model_implementations import load_model

from utils import (
    set_seed,
    resolve_feature_subdir,
    filter_pairs_by_ids,
    sample_ids,
)


set_seed(42)


CSV_FIELDS = [
    "dataset",
    "group",
    "extractor",
    "feature_key",
    "manipulation",
    "model_type",
    "original_id",
    "delta_norm",
    "relative_delta",
    "perturbation_type",
    "perturbation_multiplier",
    "delta_source",
    "metric_name",
    "metric_value",
    "feature_shape",
]


def cleanup_cuda(*objects, synchronize=True):
    """
    Delete references, run Python garbage collection, and release unused
    PyTorch CUDA cache blocks.
    """
    for obj in objects:
        del obj

    gc.collect()

    if torch.cuda.is_available():
        if synchronize:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _ensure_pairs_2d(pairs):
    pairs = np.asarray(pairs, dtype=object)

    if pairs.ndim == 1 and len(pairs) > 0:
        try:
            pairs = np.stack(pairs)
        except ValueError:
            pairs = np.array(
                [np.asarray(pair, dtype=object) for pair in pairs],
                dtype=object,
            )

    return pairs


class SwinV2StagedFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        device: torch.device | str = "cpu",
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
        )

        if not isinstance(self.backbone, SwinTransformerV2):
            raise TypeError(
                f"Expected SwinTransformerV2, got {type(self.backbone)}"
            )

        self.eval().to(device)

    def forward_features(self, x: torch.Tensor, stage: int):
        x = self.backbone.patch_embed(x)

        for index, layer in enumerate(self.backbone.layers):
            x = layer(x)

            if index == stage:
                return self._to_nchw(x)

        return self._to_nchw(x)

    @staticmethod
    def _to_nchw(x: torch.Tensor):
        if x.ndim == 3:
            batch_size, num_tokens, channels = x.shape
            height = width = int(num_tokens**0.5)

            if height * width != num_tokens:
                raise ValueError(
                    f"Cannot reshape {num_tokens} tokens into a square map"
                )

            x = x.transpose(1, 2).reshape(
                batch_size,
                channels,
                height,
                width,
            )

        elif x.ndim == 4:
            _, dim1, dim2, dim3 = x.shape

            # timm SwinV2 commonly returns NHWC.
            if dim3 > dim1 and dim3 > dim2:
                x = x.permute(0, 3, 1, 2)

        else:
            raise RuntimeError(
                f"Unexpected SwinV2 feature shape: {tuple(x.shape)}"
            )

        return x.contiguous()

    def forward(self, x):
        return self.backbone(x)


def load_pairs_and_meta(
    extractor_name,
    group_name,
    dataset,
    dataset_path,
    feature_subdir,
    manipulation,
):
    if dataset == "CUB_200_2011":
        base_root = os.path.join(
            dataset_path,
            "CUB_200_2011",
            "CUB_200_2011",
        )
    elif dataset == "STANFORD_CARS":
        base_root = os.path.join(dataset_path, "STANFORD_CARS")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    feature_dir = os.path.join(
        base_root,
        f"{extractor_name}_features",
        "augmented_test",
        feature_subdir,
    )

    meta_path = os.path.join(feature_dir, "meta.jsonl")

    row_to_meta = {}

    with open(meta_path) as file:
        for line in file:
            metadata = json.loads(line)
            row_to_meta[int(metadata["row_idx"])] = metadata

    pairs_path = os.path.join(feature_dir, "pairs.npy")
    pairs = np.load(pairs_path, allow_pickle=True)
    pairs = _ensure_pairs_2d(pairs)

    canonical_pairs = []

    for pair in pairs:
        if len(pair) < 2:
            continue

        original_idx = int(pair[0])
        target_idx = int(pair[1])

        if target_idx not in row_to_meta:
            raise KeyError(
                f"Target row {target_idx} is missing from {meta_path}"
            )

        target_manipulation = row_to_meta[target_idx]["manipulation"]

        canonical_pairs.append(
            (
                original_idx,
                target_idx,
                target_manipulation,
            )
        )

    if canonical_pairs:
        pairs = np.asarray(canonical_pairs, dtype=object)
    else:
        pairs = np.empty((0, 3), dtype=object)

    if manipulation is not None:
        pairs = np.asarray(
            [
                pair
                for pair in pairs
                if pair[2] == manipulation
            ],
            dtype=object,
        )

        if pairs.size == 0:
            pairs = np.empty((0, 3), dtype=object)

    return pairs, row_to_meta, feature_dir


def get_backbone(name, device):
    if name == "convnext":
        return (
            timm.create_model(
                "convnext_base.fb_in22k_ft_in1k",
                pretrained=True,
                features_only=True,
            )
            .eval()
            .to(device)
        )

    if name == "swinv2":
        return SwinV2StagedFeatureExtractor(
            "swinv2_base_window12to24_192to384_22kft1k",
            pretrained=True,
            device=device,
        )

    if name == "dinov3":
        return DINOv3FeatureExtractor(
            model_name=DEFAULT_MODEL_NAME,
            hf_token=os.environ.get("HF_TOKEN"),
            device=device,
        )

    raise ValueError(f"Unknown backbone name: {name}")


def initialize_reconstructor(extractor, feature_key, device):
    backbone = get_backbone(extractor, device)

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
        raise ValueError(f"Unknown extractor: {extractor}")

    if hasattr(reconstructor, "eval"):
        reconstructor.eval()

    return backbone, reconstructor


def run_eval_loop(test_loader, step_fn):
    with torch.inference_mode():
        for batch in test_loader:
            step_fn(batch)


def build_random_perturbation(original_feat, target_norm):
    noise = torch.randn_like(original_feat)
    noise_norm = torch.linalg.vector_norm(noise.reshape(-1), ord=2)

    if noise_norm.item() == 0:
        return original_feat.clone()

    return original_feat + noise * (target_norm / noise_norm)


def feature_norm(tensor):
    return torch.linalg.vector_norm(
        tensor.reshape(-1),
        ord=2,
    ).item()


def write_control_record(csv_path, record):
    exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(record)


def compute_baseline_and_perturbation_metrics(
    original_feat,
    target_feat,
    mapped_feat,
    reconstructor,
    evaluator,
    info,
    save_csv,
    normalized=False,
    normalizer=None,
):
    """
    original_feat, target_feat, and mapped_feat are CPU tensors here.
    """
    if normalized and normalizer is not None:
        original_feat = normalizer.denormalize(original_feat)
        target_feat = normalizer.denormalize(target_feat)
        mapped_feat = normalizer.denormalize(mapped_feat)

    original_np = original_feat.numpy()
    mapped_np = mapped_feat.numpy()

    with torch.inference_mode():
        original_img = process_features(
            original_np,
            reconstructor,
        )
        mapped_img = process_features(
            mapped_np,
            reconstructor,
        )

    image_dir = os.path.join(
        os.path.dirname(save_csv),
        "reconstruction_control_images",
        info["dataset"],
        info["group"],
        info["extractor"],
        info["feature_key"],
        info["manipulation"],
        info["model_type"],
        info["original_id"],
    )

    os.makedirs(image_dir, exist_ok=True)

    original_img.save(
        os.path.join(image_dir, "original.png")
    )
    mapped_img.save(
        os.path.join(image_dir, "mapped.png")
    )

    delta_gt = feature_norm(target_feat - original_feat)
    delta_map = feature_norm(mapped_feat - original_feat)
    original_norm = feature_norm(original_feat)

    relative_delta_gt = delta_gt / (original_norm + 1e-12)
    relative_delta_map = delta_map / (original_norm + 1e-12)

    mapped_record = {
        **info,
        "delta_norm": delta_map,
        "relative_delta": relative_delta_map,
        "perturbation_type": "mapped",
        "perturbation_multiplier": 1.0,
        "delta_source": "map",
        "feature_shape": tuple(original_feat.shape),
    }

    mapped_metrics = evaluator.calc_metrics(
        original_img,
        mapped_img,
    )

    for name, value in mapped_metrics.items():
        record = mapped_record.copy()
        record["metric_name"] = name
        record["metric_value"] = value
        write_control_record(save_csv, record)

    for multiplier in (0.5, 1.0, 2.0):
        target_norm = delta_gt * multiplier

        random_feat = build_random_perturbation(
            original_feat,
            target_norm,
        )

        random_img = process_features(
            random_feat.numpy(),
            reconstructor,
        )

        random_img.save(
            os.path.join(
                image_dir,
                f"random_gt_{multiplier:.1f}.png",
            )
        )

        random_metrics = evaluator.calc_metrics(
            original_img,
            random_img,
        )

        random_record = {
            **info,
            "delta_norm": delta_gt,
            "relative_delta": relative_delta_gt,
            "perturbation_type": "random",
            "perturbation_multiplier": multiplier,
            "delta_source": "gt",
            "feature_shape": tuple(original_feat.shape),
        }

        for name, value in random_metrics.items():
            record = random_record.copy()
            record["metric_name"] = name
            record["metric_value"] = value
            write_control_record(save_csv, record)

        del random_feat, random_img, random_metrics


def get_original_id(meta, row_idx):
    row_idx = int(row_idx)

    if row_idx not in meta:
        raise KeyError(
            f"Row index {row_idx} not present in metadata"
        )

    return str(meta[row_idx]["original_id"]).strip()


def run_one_model(
    *,
    test_loader,
    model,
    reconstructor,
    evaluator,
    device,
    meta,
    dataset,
    group_name,
    extractor,
    feature_key,
    manipulation,
    model_type,
    control_csv,
    normalized,
    normalizer,
):
    model.eval()

    def step(batch):
        (
            orig_feats,
            target_feats,
            orig_labels,
            target_labels,
            manips,
            orig_indices,
            target_indices,
        ) = batch

        del orig_labels
        del target_labels
        del manips
        del target_indices

        orig_feats = orig_feats.to(
            device,
            non_blocking=True,
        )
        target_feats = target_feats.to(
            device,
            non_blocking=True,
        )

        mapped_feats = model(orig_feats)

        for sample_index, row_idx in enumerate(orig_indices):
            row_idx = int(row_idx)
            original_id = get_original_id(meta, row_idx)

            info = {
                "dataset": dataset,
                "group": group_name,
                "extractor": extractor,
                "feature_key": feature_key,
                "manipulation": manipulation,
                "model_type": model_type,
                "original_id": original_id,
            }

            # Move only the current sample to CPU. The reconstruction model
            # should ideally also be on CPU.
            original_cpu = (
                orig_feats[sample_index]
                .detach()
                .cpu()
            )
            target_cpu = (
                target_feats[sample_index]
                .detach()
                .cpu()
            )
            mapped_cpu = (
                mapped_feats[sample_index]
                .detach()
                .cpu()
            )

            compute_baseline_and_perturbation_metrics(
                original_feat=original_cpu,
                target_feat=target_cpu,
                mapped_feat=mapped_cpu,
                reconstructor=reconstructor,
                evaluator=evaluator,
                info=info,
                save_csv=control_csv,
                normalized=normalized,
                normalizer=normalizer,
            )

            del original_cpu
            del target_cpu
            del mapped_cpu

        del mapped_feats
        del orig_feats
        del target_feats

    run_eval_loop(test_loader, step)

    # step closes over model/reconstructor. Delete it explicitly.
    del step


def run_control_experiment(
    config,
    dataset,
    group_name,
    group_cfg,
    manipulation,
    extractor,
    pairs,
    meta,
    root,
    selected_ids,
    mapping_device,
    reconstruction_device,
    model_path,
    norm_path,
    eval_path,
):
    del selected_ids

    feature_subdir = resolve_feature_subdir(group_name)
    feature_cfgs = config["feature_keys"][extractor]

    extractor_eval_path = os.path.join(
        eval_path,
        extractor,
    )
    os.makedirs(extractor_eval_path, exist_ok=True)

    evaluator = Evaluator(
        eval_path,
        extractor,
    )

    num_workers = 2
    batch_size = int(config.get("batch_size", 1))

    for feature_key, feat_cfg in feature_cfgs.items():
        model = None
        backbone = None
        reconstructor = None
        test_loader = None
        base_dataset = None
        normalizer = None

        try:
            input_dim = feat_cfg["input_dim"]
            output_dim = feat_cfg["output_dim"]
            spatial_size = feat_cfg["spatial_size"]
            num_vec = spatial_size * spatial_size

            normalized = bool(
                feat_cfg.get(
                    "normalize_features",
                    True,
                )
            )

            print(
                f"\nStarting {dataset} | {group_name} | "
                f"{manipulation} | {extractor} | {feature_key}"
            )

            backbone, reconstructor = initialize_reconstructor(
                extractor=extractor,
                feature_key=feature_key,
                device=reconstruction_device,
            )
        
            base_dataset = PairedFeatureDataset(
                path_prefix=root,
                feature_key=feature_key,
                pairs=pairs,
                manipulation=manipulation,
            )

            if normalized:
                normalizer = DatasetNormalize(
                    base_dataset,
                    dataset,
                    "resized",
                    feature_key,
                    norm_path,
                    config.get(
                        "recalc_norm_params",
                        False,
                    ),
                )

            collate_fn = lambda batch: test_collate_fn(
                batch,
                normalizer=normalizer,
            )

            loader_kwargs = {
                "dataset": base_dataset,
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": num_workers,
                "collate_fn": collate_fn,
                "pin_memory": (
                    mapping_device.type == "cuda"
                    and num_workers >= 0
                ),
            }

            if num_workers > 0:
                loader_kwargs["persistent_workers"] = False
                loader_kwargs["prefetch_factor"] = 1

            test_loader = DataLoader(**loader_kwargs)

            # Peek only when required for logging. With num_workers=0 this
            # does not create persistent worker processes.
            feature_shape = None
            peek_iter = None
            peek_batch = None

            try:
                peek_iter = iter(test_loader)
                peek_batch = next(peek_iter)

                orig_feats, target_feats, *_ = peek_batch
                feature_shape = tuple(orig_feats[0].shape)

            except StopIteration:
                feature_shape = None

            finally:
                del peek_iter
                del peek_batch
                gc.collect()

            if feature_shape is None:
                print(
                    f"Control experiment | {extractor} | "
                    f"{feature_key} | {manipulation} | no batches"
                )
                continue

            print(
                f"Control experiment | {extractor} | "
                f"{feature_key} | {manipulation} | "
                f"feature shape {feature_shape}"
            )

            for model_type in group_cfg.get("models", []):
                model = None

                try:
                    model = load_model(
                        dataset=dataset,
                        extractor_name=extractor,
                        model_path=model_path,
                        normalized=normalized,
                        feature_subdir=feature_subdir,
                        target_manipulation=manipulation,
                        model_type=model_type,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_params=config.get(
                            "models",
                            {},
                        ).get(
                            model_type,
                            {},
                        ),
                        device=mapping_device,
                        feature_key=feature_key,
                        loss_name=config.get(
                            "loss_functions",
                            ["mse"],
                        )[0],
                        num_feature_vectors=num_vec,
                        apply_transform=group_cfg.get(
                            "apply_transform",
                            False,
                        ),
                    )

                    if model is None:
                        print(
                            f"Skipping missing checkpoint for "
                            f"{model_type} {extractor} "
                            f"{feature_key} {manipulation}"
                        )
                        continue

                    model.eval()

                    control_csv = os.path.join(
                        extractor_eval_path,
                        f"{feature_subdir}_{dataset}_"
                        f"{feature_key}_{manipulation}_"
                        f"{model_type}_"
                        f"reconstruction_control.csv",
                    )

                    print(
                        f"Writing control metrics to: {control_csv}"
                    )

                    run_one_model(
                        test_loader=test_loader,
                        model=model,
                        reconstructor=reconstructor,
                        evaluator=evaluator,
                        device=mapping_device,
                        meta=meta,
                        dataset=dataset,
                        group_name=group_name,
                        extractor=extractor,
                        feature_key=feature_key,
                        manipulation=manipulation,
                        model_type=model_type,
                        control_csv=control_csv,
                        normalized=normalized,
                        normalizer=normalizer,
                    )

                finally:
                    if model is not None:
                        model.cpu()

                    del model
                    model = None

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()


        finally:
            # Release the DataLoader before its Dataset.
            del test_loader
            del normalizer
            del base_dataset

            # Delete the reconstructor and backbone as a pair.
            del reconstructor
            del backbone

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()



def ids_for_extractor(pairs, meta):
    ids = set()

    for pair in pairs:
        if len(pair) < 2:
            continue

        original_idx = int(pair[0])

        if original_idx not in meta:
            continue

        original_id = str(
            meta[original_idx]["original_id"]
        ).strip()

        ids.add(original_id)

    return ids


def get_shared_ids_all_extractors(extractor_data):
    shared_ids = None

    for _, pairs, meta, _ in extractor_data:
        current_ids = ids_for_extractor(pairs, meta)

        if shared_ids is None:
            shared_ids = current_ids
        else:
            shared_ids &= current_ids

    return sorted(shared_ids or [])


def main():
    start = time.time()

    with open("../config/test_mapping_control.yaml") as file:
        config = yaml.safe_load(file)

    mapping_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Prefer CPU for the reconstruction model to avoid keeping the mapping
    # model, backbone, and FeatInv model on the GPU simultaneously.
    reconstruction_device = torch.device(
        config.get(
            "reconstruction_device",
            "cpu",
        )
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

    subset_size = config.get("subset_size")

    allowed_extractors = config.get(
        "extractor_name",
        [
            "convnext",
            "swinv2",
            "dinov3",
        ],
    )

    for dataset, groups in config["target_manipulations"].items():
        for group_name, group_cfg in groups.items():
            feature_subdir = resolve_feature_subdir(
                group_name
            )

            available_extractors = []

            # Only use this pass to determine which extractors have data.
            for extractor in allowed_extractors:
                try:
                    _, _, _ = load_pairs_and_meta(
                        extractor_name=extractor,
                        group_name=group_name,
                        dataset=dataset,
                        dataset_path=dataset_path,
                        feature_subdir=feature_subdir,
                        manipulation=group_cfg["manipulations"][0],
                    )
                except Exception as error:
                    print(
                        f"Skipping unavailable extractor "
                        f"{extractor}: {error}"
                    )
                    continue

                available_extractors.append(extractor)

            for manipulation in group_cfg["manipulations"]:
                extractor_data = []

                for extractor in available_extractors:
                    try:
                        pairs, meta, root = load_pairs_and_meta(
                            extractor_name=extractor,
                            group_name=group_name,
                            dataset=dataset,
                            dataset_path=dataset_path,
                            feature_subdir=feature_subdir,
                            manipulation=manipulation,
                        )
                    except Exception as error:
                        print(
                            f"Skipping {extractor} for "
                            f"{dataset} | {group_name} | "
                            f"{manipulation}: {error}"
                        )
                        continue

                    extractor_data.append(
                        (
                            extractor,
                            pairs,
                            meta,
                            root,
                        )
                    )

                if not extractor_data:
                    print(
                        f"No extractor data for "
                        f"{dataset} | {group_name} | "
                        f"{manipulation}"
                    )
                    continue

                available_ids = get_shared_ids_all_extractors(
                    extractor_data
                )

                if not available_ids:
                    print(
                        f"No shared IDs for "
                        f"{dataset} | {group_name} | "
                        f"{manipulation}"
                    )
                    continue

                requested = (
                    len(available_ids)
                    if subset_size is None
                    else min(
                        int(subset_size),
                        len(available_ids),
                    )
                )

                selected = set(
                    sample_ids(
                        available_ids,
                        requested,
                    )
                )

                print(
                    f"{dataset} | {group_name} | "
                    f"{manipulation} | "
                    f"available={len(available_ids)} | "
                    f"selected={len(selected)}"
                )

                for extractor, pairs, meta, root in extractor_data:
                    filtered_pairs = filter_pairs_by_ids(
                        pairs,
                        meta,
                        selected,
                    )

                    if len(filtered_pairs) == 0:
                        print(
                            f"No filtered pairs for "
                            f"{extractor} | {feature_subdir} | "
                            f"{manipulation}"
                        )
                        continue

                    run_control_experiment(
                        config=config,
                        dataset=dataset,
                        group_name=group_name,
                        group_cfg=group_cfg,
                        manipulation=manipulation,
                        extractor=extractor,
                        pairs=filtered_pairs,
                        meta=meta,
                        root=root,
                        selected_ids=selected,
                        mapping_device=mapping_device,
                        reconstruction_device=reconstruction_device,
                        model_path=model_path,
                        norm_path=norm_path,
                        eval_path=eval_path,
                    )

    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()