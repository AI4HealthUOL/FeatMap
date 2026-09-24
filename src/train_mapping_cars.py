import argparse
import os
import time
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model_implementations import MappingModel
from prepare_datasets.feature_dataset_impl import (
    DatasetNormalize,
    FeatureDataset,
    MappingDataModule,
)
from utils import (
    build_manip_indices,
    get_mapping_run_paths,
    resolve_feature_subdir,
    set_seed,
)


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "train_mapping_cars.yaml"

SUPPORTED_EXTRACTORS = (
    "convnext",
    "swinv2",
    "dinov3",
)

SUPPORTED_MODELS = (
    "linear",
    "additive",
    "local_nonshared_linear",
    "mlp",
    "cnn",
    "transformer",
)


torch.set_float32_matmul_precision("high")
set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train feature-space mapping models for selected "
            "extractor, feature, manipulation, and model."
        )
    )

    parser.add_argument(
        "--extractor",
        type=str,
        default=None,
        choices=SUPPORTED_EXTRACTORS,
        help=(
            "Extractor to train on. If omitted, all configured "
            "extractors are used."
        ),
    )

    parser.add_argument(
        "--feature-key",
        type=str,
        default=None,
        help=(
            "Feature key to train on, for example feat0. If omitted, "
            "all valid feature keys are used."
        ),
    )

    parser.add_argument(
        "--model",
        dest="model_type",
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help=(
            "Mapping model to train. If omitted, all models allowed "
            "by the target group are used."
        ),
    )

    parser.add_argument(
        "--target-group",
        type=str,
        default=None,
        help=(
            "Target manipulation group, for example "
            "qwen_gs_1_infsteps_10. If omitted, all groups are used."
        ),
    )

    parser.add_argument(
        "--manipulation",
        type=str,
        default=None,
        help=(
            "Specific manipulation to train. If omitted, all "
            "manipulations in the selected group are used."
        ),
    )

    parser.add_argument(
        "--loss-name",
        type=str,
        default=None,
        help=(
            "Specific loss name, for example mse+median_cos. If "
            "omitted, all configured losses are used."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run identifier printed in the log.",
    )

    return parser.parse_args()


class SubsetFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper exposing only selected indices of a base dataset.
    """

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_loss_name(loss_function):
    if isinstance(loss_function, dict):
        return loss_function.get("name", "mse")

    return str(loss_function)


def filter_loss_functions(config, loss_name_filter=None):
    loss_functions = list(config["loss_functions"])

    if loss_name_filter is None:
        return loss_functions

    filtered = [
        loss_function
        for loss_function in loss_functions
        if get_loss_name(loss_function) == loss_name_filter
    ]

    if not filtered:
        configured_names = [
            get_loss_name(loss_function)
            for loss_function in loss_functions
        ]

        raise ValueError(
            f"Loss {loss_name_filter!r} is not configured. "
            f"Available losses: {configured_names}"
        )

    return filtered


def get_filtered_manipulations(
    group_cfg,
    manipulation_filter=None,
):
    manipulations = list(group_cfg["manipulations"])

    if manipulation_filter is None:
        return manipulations

    if manipulation_filter not in manipulations:
        raise ValueError(
            f"Manipulation {manipulation_filter!r} is not configured "
            f"for this target group. Available manipulations: "
            f"{manipulations}"
        )

    return [manipulation_filter]


def get_filtered_models(group_cfg, model_filter=None):
    models = list(group_cfg["models"])

    if model_filter is None:
        return models

    if model_filter not in models:
        raise ValueError(
            f"Model {model_filter!r} is not allowed for this target "
            f"group. Available models: {models}"
        )

    return [model_filter]


def get_filtered_extractors(config, extractor_filter=None):
    extractors = list(config["extractor_name"])

    if extractor_filter is None:
        return extractors

    if extractor_filter not in extractors:
        raise ValueError(
            f"Extractor {extractor_filter!r} is not configured. "
            f"Configured extractors: {extractors}"
        )

    return [extractor_filter]


def get_feature_configs(
    config,
    extractor_name,
    group_cfg,
    feature_key_filter=None,
):
    if extractor_name not in config["feature_keys"]:
        raise KeyError(
            f"No feature configuration found for extractor "
            f"{extractor_name!r}."
        )

    feature_cfgs = dict(config["feature_keys"][extractor_name])

    if feature_key_filter is not None:
        if feature_key_filter not in feature_cfgs:
            raise ValueError(
                f"Feature key {feature_key_filter!r} is not configured "
                f"for extractor {extractor_name!r}. Available keys: "
                f"{list(feature_cfgs.keys())}"
            )

        feature_cfgs = {
            feature_key_filter: feature_cfgs[feature_key_filter]
        }

    group_feature_keys = group_cfg.get("feature_keys")

    if group_feature_keys is not None:
        feature_cfgs = {
            feature_key: feature_cfg
            for feature_key, feature_cfg in feature_cfgs.items()
            if feature_key in group_feature_keys
        }

    return feature_cfgs


def build_normalizers(
    train_feature_dir,
    feature_key,
    source_dataset,
    extractor_norm_path,
    shared_norm_type,
):
    """
    Build normalization statistics from a reproducible subset.
    """

    meta_path = os.path.join(
        train_feature_dir,
        "meta.jsonl",
    )

    base_dataset = FeatureDataset(
        path_prefix=train_feature_dir,
        feature_key=feature_key,
    )

    manip_indices = build_manip_indices(
        meta_path,
        shared_norm_type,
    )

    if not manip_indices:
        raise RuntimeError(
            f"No manipulation indices found in {meta_path}."
        )

    rng = np.random.default_rng(42)
    n_stats = min(200, len(manip_indices))

    selected_positions = rng.choice(
        len(manip_indices),
        n_stats,
        replace=False,
    )

    stats_indices = [
        manip_indices[position]
        for position in selected_positions
    ]

    manip_subset = SubsetFeatureDataset(
        base_dataset,
        stats_indices,
    )

    return DatasetNormalize(
        manip_subset,
        source_dataset,
        shared_norm_type,
        feature_key,
        extractor_norm_path,
        recalc_norm_params=True,
    )


def build_model_config(
    config,
    model_type,
    feature_key,
    input_dim,
    output_dim,
    spatial_size,
    group_cfg,
    loss_function,
):
    """
    Build the MappingModel configuration.
    """

    model_params = config["models"][model_type]

    if isinstance(model_params, dict):
        per_feature_params = model_params.get(
            feature_key,
            {},
        )

        model_config = {
            **config,
            **per_feature_params,
        }

        params_for_flags = model_params
    else:
        model_config = dict(config)
        params_for_flags = {}

    loss_name = get_loss_name(loss_function)

    if isinstance(loss_function, dict):
        loss_params = loss_function.get(
            "params",
            {},
        )
    else:
        loss_params = {}

    model_config.update({
        "feature_key": feature_key,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "spatial_size": spatial_size,
        "model_type": model_type,
        "loss_function": loss_name,
        "loss_params": loss_params,
        "apply_transform": group_cfg.get(
            "apply_transform",
            False,
        ),
        "scheduler": params_for_flags.get(
            "scheduler",
            "ReduceLROnPlateau",
        ),
    })

    return model_config


def make_safe_name(value):
    return (
        str(value)
        .replace("+", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )

def log_completed_run(
    log_path: str,
    extractor: str,
    feature_key: str,
    model_type: str,
    manipulation: str,
    loss_name: str,
    source_dataset: str,
    target_group: str,
    run_name: str | None = None,
    extra_info: dict | None = None,
) -> None:
    """
    Append a single line describing a completed training run to a central log file.
    """
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "runs_done.log")

    timestamp = datetime.now().isoformat(timespec="seconds")

    info_parts = [
        f"timestamp={timestamp}",
        f"dataset={source_dataset}",
        f"group={target_group}",
        f"extractor={extractor}",
        f"feature={feature_key}",
        f"model={model_type}",
        f"manipulation={manipulation}",
        f"loss={loss_name}",
    ]

    if run_name is not None:
        info_parts.append(f"run_name={run_name}")

    if extra_info:
        for k, v in extra_info.items():
            info_parts.append(f"{k}={v}")

    line = " | ".join(info_parts) + "\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)

def train_model(
    model_config,
    checkpoint_dir,
    log_dir,
    data_module,
    model_type,
    feature_key,
    loss_function,
    normalize_features,
    accumulate_grad_batches,
):
    model = MappingModel(model_config)

    loss_name = get_loss_name(loss_function)
    safe_loss_name = make_safe_name(loss_name)

    manipulation = model_config.get(
        "manipulation",
        "N_A",
    )

    extractor_name = model_config.get(
        "extractor_name",
        "unknown_extractor",
    )

    source_dataset = model_config.get(
        "source_dataset",
        "unknown_dataset",
    )

    target_group = model_config.get(
        "target_group",
        "unknown_group",
    )

    run_name = model_config.get(
        "run_name",
        None,
    )

    header_lines = [
        "=" * 78,
        "Training mapping model",
        f"  source_dataset      : {source_dataset}",
        f"  target_group        : {target_group}",
        f"  extractor           : {extractor_name}",
        f"  feature_key         : {feature_key}",
        f"  model_type          : {model_type}",
        f"  manipulation        : {manipulation}",
        f"  loss_function       : {loss_name}",
        f"  normalize_features  : {normalize_features}",
        (
            "  apply_transform     : "
            f"{model_config.get('apply_transform', False)}"
        ),
        f"  run_name            : {run_name}",
        f"  checkpoint_dir      : {checkpoint_dir}",
        f"  log_dir             : {log_dir}",
        "=" * 78,
    ]

    for line in header_lines:
        print(line, flush=True)

    logger_name = (
        f"{model_type}_"
        f"{feature_key}_"
        f"{make_safe_name(manipulation)}_"
        f"loss-{safe_loss_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=logger_name,
    )

    checkpoint_name = (
        f"{extractor_name}_"
        f"{model_type}_"
        f"{feature_key}_"
        f"{make_safe_name(manipulation)}_"
        f"loss-{safe_loss_name}_"
        f"normalized-{normalize_features}_"
        "mapping_model"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=False,
        every_n_epochs=2,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=model_config["patience"],
        mode="min",
        min_delta=0.001,
    )

    trainer = pl.Trainer(
        max_epochs=model_config["num_epochs"],
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
        accelerator=(
            "gpu"
            if torch.cuda.is_available()
            else "cpu"
        ),
        devices=1,
        logger=logger,
        log_every_n_steps=100,
        enable_model_summary=False,
        enable_progress_bar=True,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    
    trainer.fit(
        model,
        datamodule=data_module,
    )


def count_experiment_runs(
    config,
    extractor_filter=None,
    feature_key_filter=None,
    model_filter=None,
    target_group_filter=None,
    manipulation_filter=None,
    loss_name_filter=None,
):
    total_runs = 0

    extractors = get_filtered_extractors(
        config,
        extractor_filter,
    )

    loss_functions = filter_loss_functions(
        config,
        loss_name_filter,
    )

    manipulations_by_dataset = config["target_manipulations"]

    for source_dataset, groups in manipulations_by_dataset.items():
        for group_name, group_cfg in groups.items():
            if (
                target_group_filter is not None
                and group_name != target_group_filter
            ):
                continue

            models = get_filtered_models(
                group_cfg,
                model_filter,
            )

            manipulations = get_filtered_manipulations(
                group_cfg,
                manipulation_filter,
            )

            for extractor_name in extractors:
                feature_cfgs = get_feature_configs(
                    config,
                    extractor_name,
                    group_cfg,
                    feature_key_filter,
                )

                total_runs += (
                    len(manipulations)
                    * len(feature_cfgs)
                    * len(models)
                    * len(loss_functions)
                )

    return total_runs

def should_skip_combination(
    extractor_name: str,
    feature_key: str,
    model_type: str,
    skip_combinations,
) -> bool:
    if not skip_combinations:
        return False
    for combo in skip_combinations:
        ex, fk, mt = combo
        if extractor_name == ex and feature_key == fk and model_type == mt:
            return True
    return False

def run_experiment(
    config,
    extractor_filter=None,
    feature_key_filter=None,
    model_filter=None,
    target_group_filter=None,
    manipulation_filter=None,
    loss_name_filter=None,
    run_name=None,
):
    manipulations_by_dataset = config["target_manipulations"]

    model_path = os.path.expandvars(config["model_path"])
    log_path = os.path.expandvars(config["log_path"])
    norm_params_path = os.path.expandvars(config["norm_params_path"])
    dataset_path = os.path.expandvars(config["dataset_path"])

    shared_norm_type = config.get("shared_norm_type", "resized")
    accumulate_grad_batches = config.get("accumulate_grad_batches", 2)

    # Read skip combinations (default to empty list if not present)
    skip_combinations = config.get("skip_combinations", [])

    os.makedirs(model_path, exist_ok=True)

    total_runs = count_experiment_runs(
        config,
        extractor_filter=extractor_filter,
        feature_key_filter=feature_key_filter,
        model_filter=model_filter,
        target_group_filter=target_group_filter,
        manipulation_filter=manipulation_filter,
        loss_name_filter=loss_name_filter,
    )

    if total_runs == 0:
        raise RuntimeError(
            "No experiments match the requested filters."
        )

    print(
        f"Total matching runs: {total_runs}",
        flush=True,
    )

    run_idx = 0

    extractors = get_filtered_extractors(
        config,
        extractor_filter,
    )

    loss_functions = filter_loss_functions(
        config,
        loss_name_filter,
    )

    for source_dataset, groups in manipulations_by_dataset.items():
        root_dir = os.path.join(
            dataset_path,
            source_dataset,
        )

        for group_name, group_cfg in groups.items():
            if (
                target_group_filter is not None
                and group_name != target_group_filter
            ):
                continue

            feature_subdir = resolve_feature_subdir(
                group_name
            )

            apply_flag = group_cfg.get(
                "apply_transform",
                False,
            )

            models = get_filtered_models(
                group_cfg,
                model_filter,
            )

            manipulations = get_filtered_manipulations(
                group_cfg,
                manipulation_filter,
            )

            for extractor_name in extractors:
                train_feature_dir = os.path.join(
                    root_dir,
                    f"{extractor_name}_features",
                    "augmented_train",
                    feature_subdir,
                )

                if not os.path.isdir(train_feature_dir):
                    raise FileNotFoundError(
                        "Feature directory does not exist: "
                        f"{train_feature_dir}"
                    )

                feature_cfgs = get_feature_configs(
                    config,
                    extractor_name,
                    group_cfg,
                    feature_key_filter,
                )

                if not feature_cfgs:
                    continue

                extractor_model_path = os.path.join(
                    model_path,
                    extractor_name,
                )

                extractor_log_path = os.path.join(
                    log_path,
                    extractor_name,
                )

                extractor_norm_path = os.path.join(
                    norm_params_path,
                    extractor_name,
                )

                os.makedirs(
                    extractor_model_path,
                    exist_ok=True,
                )

                os.makedirs(
                    extractor_log_path,
                    exist_ok=True,
                )

                os.makedirs(
                    extractor_norm_path,
                    exist_ok=True,
                )

                for manipulation in manipulations:
                    for feature_key, feat_cfg in feature_cfgs.items():
                        input_dim = feat_cfg["input_dim"]
                        output_dim = feat_cfg["output_dim"]
                        spatial_size = feat_cfg["spatial_size"]
                        normalize_features = feat_cfg[
                            "normalize_features"
                        ]

                        normalizers = None

                        if normalize_features:
                            normalizers = {
                                "shared": build_normalizers(
                                    train_feature_dir,
                                    feature_key,
                                    source_dataset,
                                    extractor_norm_path,
                                    shared_norm_type,
                                )
                            }

                        pairs_path = os.path.join(
                            train_feature_dir,
                            "pairs.npy",
                        )

                        if not os.path.isfile(pairs_path):
                            raise FileNotFoundError(
                                f"Pairs file does not exist: {pairs_path}"
                            )

                        pairs = np.load(
                            pairs_path,
                            allow_pickle=True,
                        )

                        data_module = MappingDataModule(
                            feature_path=train_feature_dir,
                            feature_key=feature_key,
                            pairs=pairs,
                            config=config,
                            normalizers=normalizers,
                            manipulation=manipulation,
                        )

                        for model_type in models:
                            if should_skip_combination(
                                extractor_name,
                                feature_key,
                                model_type,
                                skip_combinations,
                            ):
                                continue

                            for loss_function in loss_functions:
                                run_idx += 1
                                loss_name = get_loss_name(loss_function)

                                print(
                                    f"[{run_idx}/{total_runs}] "
                                    "Starting run: "
                                    f"dataset={source_dataset}, "
                                    f"group={group_name}, "
                                    f"extractor={extractor_name}, "
                                    f"feature={feature_key}, "
                                    f"model={model_type}, "
                                    f"manipulation={manipulation}, "
                                    f"loss={loss_name}",
                                    flush=True,
                                )

                                model_config = build_model_config(
                                    config=config,
                                    model_type=model_type,
                                    feature_key=feature_key,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    spatial_size=spatial_size,
                                    group_cfg=group_cfg,
                                    loss_function=loss_function,
                                )

                                model_config.update({
                                    "manipulation": manipulation,
                                    "extractor_name": extractor_name,
                                    "source_dataset": source_dataset,
                                    "target_group": group_name,
                                    "run_name": run_name,
                                })

                                base_path = get_mapping_run_paths(
                                    model_path,
                                    extractor_name,
                                    feature_subdir,
                                    source_dataset,
                                    model_type,
                                    manipulation,
                                    apply_flag,
                                )

                                loss_dir = os.path.join(
                                    base_path,
                                    feature_key,
                                    f"loss-{make_safe_name(loss_name)}",
                                )

                                checkpoint_dir = loss_dir
                                model_dir = loss_dir

                                group_type = "applied" if apply_flag else "learned"

                                log_dir = os.path.join(
                                    extractor_log_path,
                                    feature_subdir,
                                    source_dataset,
                                    group_type,
                                    model_type,
                                    manipulation,
                                    feature_key,
                                    f"loss-{make_safe_name(loss_name)}",
                                )

                                os.makedirs(model_dir, exist_ok=True)
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                os.makedirs(log_dir, exist_ok=True)

                                # --- Error handling for a single run ---
                                try:
                                    train_model(
                                        model_config=model_config,
                                        checkpoint_dir=checkpoint_dir,
                                        log_dir=log_dir,
                                        data_module=data_module,
                                        model_type=model_type,
                                        feature_key=feature_key,
                                        loss_function=loss_function,
                                        normalize_features=normalize_features,
                                        accumulate_grad_batches=accumulate_grad_batches,
                                    )

                                    # Log success
                                    extra_info = {}
                                    if "SLURM_JOB_ID" in os.environ:
                                        extra_info["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
                                    if "SLURM_ARRAY_TASK_ID" in os.environ:
                                        extra_info["slurm_array_task_id"] = os.environ["SLURM_ARRAY_TASK_ID"]

                                    log_completed_run(
                                        log_path=log_path,
                                        extractor=extractor_name,
                                        feature_key=feature_key,
                                        model_type=model_type,
                                        manipulation=manipulation,
                                        loss_name=loss_name,
                                        source_dataset=source_dataset,
                                        target_group=group_name,
                                        run_name=run_name,
                                        extra_info={**extra_info, "status": "ok"},
                                    )

                                except Exception as e:
                                    # Log failure
                                    extra_info = {
                                        "status": "failed",
                                        "error": repr(e),
                                    }
                                    if "SLURM_JOB_ID" in os.environ:
                                        extra_info["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
                                    if "SLURM_ARRAY_TASK_ID" in os.environ:
                                        extra_info["slurm_array_task_id"] = os.environ["SLURM_ARRAY_TASK_ID"]

                                    log_completed_run(
                                        log_path=log_path,
                                        extractor=extractor_name,
                                        feature_key=feature_key,
                                        model_type=model_type,
                                        manipulation=manipulation,
                                        loss_name=loss_name,
                                        source_dataset=source_dataset,
                                        target_group=group_name,
                                        run_name=run_name,
                                        extra_info=extra_info,
                                    )

                                    # Optionally print a short message; then continue to next run
                                    print(
                                        f"Run failed (logged): extractor={extractor_name}, "
                                        f"feature={feature_key}, model={model_type}, "
                                        f"manipulation={manipulation}, loss={loss_name}, "
                                        f"error={e}",
                                        flush=True,
                                    )
                                    continue  # go to next (model_type, loss_function)
                                # ---------------------------------------


if __name__ == "__main__":
    start_time = time.time()

    config = load_config()
    args = parse_args()

    if args.run_name is not None:
        print(
            f"Run name: {args.run_name}",
            flush=True,
        )

    run_experiment(
        config=config,
        extractor_filter=args.extractor,
        feature_key_filter=args.feature_key,
        model_filter=args.model_type,
        target_group_filter=args.target_group,
        manipulation_filter=args.manipulation,
        loss_name_filter=args.loss_name,
        run_name=args.run_name,
    )

    print(
        f"Total time: {time.time() - start_time:.2f}s",
        flush=True,
    )