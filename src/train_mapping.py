from model_implementations import (
    MappingModel
)
from prepare_datasets.feature_dataset_impl import (
    FeatureDataset,
    PairedFeatureDataset,
    DatasetNormalize,
    MappingDataModule,
)
from utils import resolve_feature_subdir, build_manip_indices, set_seed, get_mapping_run_paths

import os
import yaml
import torch
import time
from datetime import datetime
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")

set_seed(42)


"""
Training pipeline for feature-space mapping models.

This script trains neural networks that map feature representations
from original images to their manipulated counterparts.

Pipeline:
1. Load precomputed backbone features (ConvNeXt/SwinV2)
2. Construct paired datasets (original + manipulated)
3. Optionally normalize feature distributions
4. Train mapping models using supervised losses
5. Save best checkpoints based on validation loss

See config/train_mapping.yaml for full configuration.
"""

class SubsetFeatureDataset(torch.utils.data.Dataset):
    """
    Wrapper to create a subset view of a FeatureDataset.

    Used primarily for estimating normalization statistics on a subset
    of samples to reduce computational cost.

    Args:
        base_dataset (Dataset): Full dataset.
        indices (List[int]): Indices to include in subset.
    """

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

def load_config():
    with open("../config/train_mapping.yaml") as f:
        return yaml.safe_load(f)

def build_normalizers(train_feature_dir, feature_key, source_dataset, extractor_norm_path, shared_norm_type):
    """
    Builds normalization module if required.
    """

    meta_path = os.path.join(train_feature_dir, "meta.jsonl")

    base_dataset_full = FeatureDataset(
        path_prefix=train_feature_dir,
        feature_key=feature_key,
    )

    manip_indices = build_manip_indices(meta_path, shared_norm_type)

    rng = np.random.default_rng(42)
    n_stats = min(200, len(manip_indices))
    idx = rng.choice(len(manip_indices), n_stats, replace=False)
    stats_indices = [manip_indices[i] for i in idx]

    manip_subset = SubsetFeatureDataset(base_dataset_full, stats_indices)

    return DatasetNormalize(
        manip_subset,
        source_dataset,
        shared_norm_type,
        feature_key,
        extractor_norm_path,
        recalc_norm_params=True,
    )

def build_model_config(config, model_type, feature_key, input_dim, output_dim, spatial_size, group_cfg, loss_function):
    model_params = config["models"][model_type]

    if isinstance(model_params, dict):
        per_feat_params = model_params.get(feature_key, {})
        model_config = {**config, **per_feat_params}
        params_for_flags = model_params
    else:
        model_config = dict(config)
        params_for_flags = {}

    model_config.update({
        "feature_key": feature_key,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "spatial_size": spatial_size,
        "model_type": model_type,
        "loss_function": loss_function,
        "apply_transform": group_cfg.get("apply_transform", False),
        "scheduler": params_for_flags.get("scheduler", "ReduceLROnPlateau"),
    })

    return model_config

def train_model(
    model_config,
    model_dir,
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

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=f"{model_type}_{feature_key}_loss-{loss_function}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    apply_transform = model_config.get("apply_transform", False)
    manipulation = model_config.get("manipulation")

    if loss_function == "MSE":
        fn = f"{model_type}_{feature_key}_loss-{loss_function}_normalized_{normalize_features}_mapping_model"
    else:
        fn = f"{model_type}_{feature_key}_normalized_{normalize_features}_mapping_model"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=f"{model_type}_{feature_key}_normalized_{normalize_features}_mapping_model",
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
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        log_every_n_steps=100,
        enable_model_summary=False,
        enable_progress_bar=True,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=data_module)

def run_experiment(config):
    manipulations = config["target_manipulations"]
    source_datasets = manipulations.keys()

    model_path = os.path.expandvars(config["model_path"])
    log_path = os.path.expandvars(config["log_path"])
    norm_params_path = os.path.expandvars(config["norm_params_path"])
    dataset_path = os.path.expandvars(config["dataset_path"])
    shared_norm_type = config.get("shared_norm_type", "resized")
    accumulate_grad_batches = config.get("accumulate_grad_batches", 2)

    os.makedirs(model_path, exist_ok=True)

    for source_dataset in source_datasets:

        if source_dataset == "CUB_200_2011":
            root_dir = os.path.join(dataset_path, "CUB_200_2011/CUB_200_2011/")
        elif source_dataset == "STANFORD_CARS":
            root_dir = os.path.join(dataset_path, "STANFORD_CARS/")

        manipulation_groups = manipulations[source_dataset]

        for group_name, group_cfg in manipulation_groups.items():

            allowed_models = group_cfg["models"]
            manip_list = group_cfg["manipulations"]
            apply_flag = group_cfg.get("apply_transform", False)

            feature_subdir = resolve_feature_subdir(group_name)

            for extractor_name in config["extractor_name"]:

                train_feature_dir = os.path.join(
                    root_dir,
                    f"{extractor_name}_features",
                    "augmented_train",
                    feature_subdir
                )

                feature_cfgs = config["feature_keys"][extractor_name]

                extractor_model_path = os.path.join(model_path, extractor_name)
                extractor_log_path = os.path.join(log_path, extractor_name)
                extractor_norm_path = os.path.join(norm_params_path, extractor_name)

                os.makedirs(extractor_model_path, exist_ok=True)
                os.makedirs(extractor_log_path, exist_ok=True)
                os.makedirs(extractor_norm_path, exist_ok=True)

                for manipulation in manip_list:

                    for feature_key, feat_cfg in feature_cfgs.items():

                        input_dim = feat_cfg["input_dim"]
                        output_dim = feat_cfg["output_dim"]
                        spatial_size = feat_cfg["spatial_size"]
                        normalize_features = feat_cfg["normalize_features"]

                        normalizers = None
                        if normalize_features:
                            normalizers = {
                                "shared": build_normalizers(
                                    train_feature_dir,
                                    feature_key,
                                    source_dataset,
                                    extractor_norm_path,
                                    shared_norm_type
                                )
                            }

                        data_module = MappingDataModule(
                            feature_path=train_feature_dir,
                            feature_key=feature_key,
                            pairs=np.load(
                                os.path.join(train_feature_dir, "pairs.npy"),
                                allow_pickle=True
                            ),
                            config=config,
                            normalizers=normalizers,
                            manipulation=manipulation
                        )

                        for model_type in allowed_models:
                            for loss_function in config["loss_functions"]:

                                model_config = build_model_config(
                                    config,
                                    model_type,
                                    feature_key,
                                    input_dim,
                                    output_dim,
                                    spatial_size,
                                    group_cfg,
                                    loss_function
                                )
                                model_config["manipulation"] = manipulation 

                                base_path = get_mapping_run_paths(
                                    model_path,  
                                    extractor_name,                                  
                                    feature_subdir,
                                    source_dataset,
                                    model_type,
                                    manipulation,
                                    apply_flag
                                )

                                model_dir = base_path
                                checkpoint_dir = base_path

                                group_type = "applied" if apply_flag else "learned"

                                log_dir = os.path.join(
                                    extractor_log_path,
                                    feature_subdir,
                                    source_dataset,
                                    group_type,
                                    model_type,
                                    manipulation,
                                )

                                os.makedirs(model_dir, exist_ok=True)

                                train_model(
                                    model_config,
                                    model_dir,
                                    checkpoint_dir,
                                    log_dir,
                                    data_module,
                                    model_type,
                                    feature_key,
                                    loss_function,
                                    normalize_features,
                                    accumulate_grad_batches,
                                )
                                
# Training loop hierarchy
#
# We iterate over:
# 1. source_dataset: e.g., STANFORD_CARS
# 2. group_name: manipulation source (e.g., "direct", "qwen")
# 3. extractor_name: backbone model (convnext, swinv2)
# 4. manipulation: specific transformation (grayscale, hue_shift_60, noise_40, ...)
# 5. feature_key: feature layer (e.g., feat1, feat3)
# 6. model_type: mapping architecture (linear, transformer, ...)
# 7. loss_function: training objective
#
# This results in training a separate model for each combination.

if __name__ == "__main__":
    start_time = time.time()

    config = load_config()
    run_experiment(config)

    print(f"Total time: {time.time() - start_time:.2f}s")