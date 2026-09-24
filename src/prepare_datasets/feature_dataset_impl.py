import h5py
import os
import time
import json
import torch
import re
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*not writable.*")

"""
Implementation of writing and loading the feature datasets alongside their metadata for fast access

Supports:
- Efficient storing and loading with numpy.memmap
- memmap allows for zero-copy access, meaning it does not need to load the whole file at a time
- For the large Feature dimensions of up to 128x72x72 this still is a bottleneck on the dataloader during training
- As of now building the datasets like this proved to be the most usable
- Storing metadata (class labels, paths to the corresponding images, pairs for original vs target manipulation processing)
- Normalization

 index structure:
 {
   original_id: {
       "resized": row_idx_original,
       "rotation_15": row_idx_manipulated,
       "noise_20": row_idx_manipulated,
       ...
   }
 }

 This enables fast lookup of all manipulations for a given original image.
"""

class MemmapFeatureWriter:
    def __init__(
        self,
        path_prefix,
        feature_shapes,
        max_samples,
        dtype=np.float32,
        meta_flush_every=1000,
    ):
        os.makedirs(path_prefix, exist_ok=True)

        self.ptr = 0
        self.max_samples = max_samples
        self.meta_flush_every = meta_flush_every
        self._meta_buffer = []

        self.features = {}
        self.index = {}

        self.meta_file = open(os.path.join(path_prefix, "meta.jsonl"), "w")

        for name, shape in feature_shapes.items():
            path = os.path.join(path_prefix, f"{name}.dat")
            self.features[name] = np.memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=(max_samples, *shape),
            )

        self.targets = np.memmap(
            os.path.join(path_prefix, "target.dat"),
            mode="w+",
            dtype=np.int64,
            shape=(max_samples,),
        )

        with open(os.path.join(path_prefix, "shapes.json"), "w") as f:
            json.dump(feature_shapes, f)

    def _flush_meta_if_needed(self):
        if len(self._meta_buffer) >= self.meta_flush_every:
            for line in self._meta_buffer:
                self.meta_file.write(line)
            self._meta_buffer.clear()
            self.meta_file.flush()

    def write_batch(self, features_dict, targets, paths, metadata):
        batch_size = targets.shape[0]
        start = self.ptr

        # Features & targets
        for k, v in features_dict.items():
            # v is already on CPU from the extraction script
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            self.features[k][start:start + batch_size] = v

        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        self.targets[start:start + batch_size] = targets

        # Metadata + index
        for i, meta in enumerate(metadata):
            row_idx = start + i
            oid = meta["original_id"]
            manip = meta["manipulation"]

            self.index.setdefault(oid, {})[manip] = row_idx

            self._meta_buffer.append(
                json.dumps({**meta, "row_idx": row_idx}) + "\n"
            )

        self.ptr += batch_size
        self._flush_meta_if_needed()

    def close(self, out_index_path):
        # Flush remaining meta lines
        if self._meta_buffer:
            for line in self._meta_buffer:
                self.meta_file.write(line)
            self._meta_buffer.clear()

        for f in self.features.values():
            f.flush()
        self.targets.flush()
        self.meta_file.close()

        with open(out_index_path, "w") as f:
            json.dump(self.index, f)

def build_pairs_from_index(index, orig_key="resized"):
    """
    Builds supervised training pairs from the index.

    Each pair has the form:
        (original_idx, manipulated_idx, manipulation_type)

    Example:
        (42, 105, "rotation_90")

    Meaning:
        feature[42] → feature[105]

    The "resized" key is treated as the reference (original image).
    All other manipulations are paired against it.

    "resized" corresponds to the original image after preprocessing.
    """
    pairs = []

    for oid, entry in index.items():
        if orig_key not in entry:
            continue

        o_idx = entry[orig_key]

        for manip, m_idx in entry.items():
            if manip == orig_key:
                continue

            pairs.append(
                (o_idx, m_idx, manip)
            )

    return np.asarray(pairs, dtype=object)


class _MemmapBackend:
    """
    Backend for accessing memmap feature storage.

    Provides:
    - zero-copy feature access (numpy → torch)
    - shared memory usage across dataloader workers
    - efficient indexing without loading full dataset into RAM
    """

    def __init__(self, path_prefix, feature_key):
        shape = self._load_shape(path_prefix, feature_key)

        self.features = np.memmap(
            os.path.join(path_prefix, f"{feature_key}.dat"),
            mode="r",
            dtype=np.float32,
        ).reshape(-1, *shape)

        self.targets = np.memmap(
            os.path.join(path_prefix, "target.dat"),
            mode="r",
            dtype=np.int64,
        )

        self._shape = shape

    def _load_shape(self, path_prefix, key):
        with open(os.path.join(path_prefix, "shapes.json"), "r") as f:
            return json.load(f)[key]

    def get_feat(self, idx):
        return self.features[idx]

    def get_target(self, idx):
        return self.targets[idx]

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, path_prefix, feature_key="feat0"):
        self.backend = _MemmapBackend(path_prefix, feature_key)

    def __len__(self):
        return len(self.backend.targets)

    def __getitem__(self, idx):
        return (
            self.backend.get_feat(idx),
            int(self.backend.get_target(idx)),
        )


class PairedFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset returning paired features for mapping training.

    Labels are the original class labels from the Stanford Cars dataset.

    Each sample consists of:
        (original_feature, original_label),
        (target_feature, target_label),
        manipulation_type,
        original_index,
        target_index

    """

    def __init__(self, path_prefix, feature_key, pairs, manipulation=None, normalizer=None):
            self.backend = _MemmapBackend(path_prefix, feature_key)
            self.normalizer = normalizer

            pairs = np.asarray(pairs)
            if manipulation is not None:
                pairs = pairs[pairs[:, 2] == manipulation]

            self.o_idx = pairs[:, 0].astype(np.int64)
            self.m_idx = pairs[:, 1].astype(np.int64)
            self.manip = pairs[:, 2]

    def __len__(self):
        return len(self.o_idx)

    def __getitem__(self, i):
        o = self.o_idx[i]
        m = self.m_idx[i]

        orig_feat = self.backend.get_feat(o)      
        target_feat = self.backend.get_feat(m)  

        orig_label = int(self.backend.get_target(o))
        target_label = int(self.backend.get_target(m))

        return (
            (orig_feat, orig_label),
            (target_feat, target_label),
            self.manip[i],
            int(o),
            int(m),
        )


class MappingDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for mapping model training.

    Handles:
    - filtering pairs by manipulation type
    - optional subsampling for faster experiments
    - train/validation split
    - batching and collation

    """

    def __init__(
        self,
        feature_path,
        feature_key,
        pairs,
        config,
        manipulation=None,
        normalizers=None,
        return_full_eval_meta=False,
    ):
        super().__init__()

        self.feature_path = feature_path
        self.feature_key = feature_key
        self.pairs = pairs
        self.manipulation = manipulation
        self.normalizers = normalizers

        self.config = config
        self.batch_size = config["batch_size"]
        self.val_ratio = config["val_ratio"]
        self.num_workers = config["num_workers"]
        self.persistent_workers = config["persistent_workers"]
        self.prefetch_factor = config["prefetch_factor"]
        self.pin_memory = config["pin_memory"]
        self.train_subset_size = config.get("train_subset_size", None)
        self.return_full_eval_meta = return_full_eval_meta

    def setup(self, stage=None):
        pairs = self.pairs

        if self.manipulation is not None:
            pairs = pairs[pairs[:, 2] == self.manipulation]

        if self.train_subset_size is not None:
            rng = np.random.default_rng(42)

            k = min(self.train_subset_size, len(pairs))
            idx = rng.choice(len(pairs), k, replace=False)

            pairs = pairs[idx]

        self.paired_dataset = PairedFeatureDataset(
            path_prefix=self.feature_path,
            feature_key=self.feature_key,
            pairs=pairs,
            manipulation=self.manipulation,
            normalizer=self.normalizers["shared"] if self.normalizers else None
        )

        train_size = int((1 - self.val_ratio) * len(self.paired_dataset))
        val_size = len(self.paired_dataset) - train_size

        self.paired_train_dataset, self.paired_val_dataset = random_split(
            self.paired_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        _ = self.paired_dataset[0]

    def train_dataloader(self):
        return DataLoader(
            self.paired_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.paired_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        t0 = time.time()

        batch_size = len(batch)
        sample_shape = batch[0][0][0].shape  # (C, H, W)

        orig_feats_np = np.empty((batch_size, *sample_shape), dtype=np.float32)
        target_feats_np = np.empty((batch_size, *sample_shape), dtype=np.float32)

        orig_labels = np.empty(batch_size, dtype=np.int64)
        target_labels = np.empty(batch_size, dtype=np.int64)

        for i, b in enumerate(batch):
            orig_feats_np[i] = b[0][0]
            target_feats_np[i] = b[1][0]
            orig_labels[i] = b[0][1]
            target_labels[i] = b[1][1]

        orig_feats = torch.from_numpy(orig_feats_np)
        target_feats = torch.from_numpy(target_feats_np)
        orig_labels = torch.from_numpy(orig_labels)
        target_labels = torch.from_numpy(target_labels)

        manips = [b[2] for b in batch]

        if self.normalizers and self.normalizers.get("shared"):
            norm = self.normalizers["shared"]
            orig_feats = norm.normalize_batch(orig_feats)
            target_feats = norm.normalize_batch(target_feats)


        if not self.return_full_eval_meta:
            return orig_feats, target_feats, manips

        return (
            (orig_feats, orig_labels),
            (target_feats, target_labels),
            manips,
        )


def test_collate_fn(batch, normalizer=None):
    orig_feats_np = np.stack([b[0][0] for b in batch], axis=0)
    target_feats_np = np.stack([b[1][0] for b in batch], axis=0)

    orig_feats = torch.from_numpy(orig_feats_np).contiguous()
    target_feats = torch.from_numpy(target_feats_np).contiguous()

    orig_labels = torch.tensor([b[0][1] for b in batch], dtype=torch.long)
    target_labels = torch.tensor([b[1][1] for b in batch], dtype=torch.long)

    manips = [b[2] for b in batch]
    orig_indices = [b[3] for b in batch]
    target_indices = [b[4] for b in batch]

    if normalizer is not None:
        orig_feats = normalizer.normalize_batch(orig_feats)
        target_feats = normalizer.normalize_batch(target_feats)

    return (
        orig_feats,
        target_feats,
        orig_labels,
        target_labels,
        manips,
        orig_indices,
        target_indices,
    )

class DatasetNormalize:
    def __init__(
        self,
        dataset,
        source_dataset,
        manip,
        feat_key,
        norm_params_path,
        recalc_norm_params,
    ):
        # dataset is only used for computing stats, not stored
        os.makedirs(norm_params_path, exist_ok=True)
        norm_params_savefile = os.path.join(
            norm_params_path, source_dataset, f"{manip}_{feat_key}.pt"
        )
        os.makedirs(os.path.dirname(norm_params_savefile), exist_ok=True)

        if recalc_norm_params or not os.path.exists(norm_params_savefile):
            n = min(200, len(dataset))

            feats = torch.stack([
                torch.from_numpy(
                    dataset[i][0][0] if isinstance(dataset[i][0], tuple) else dataset[i][0]
                )
                for i in range(n)
            ])

            self.mean = feats.mean(dim=[0, 2, 3], keepdim=True)
            self.std = feats.std(dim=[0, 2, 3], keepdim=True) + 1e-6

            state = {
                "mean": self.mean,
                "std": self.std,
                "source_dataset": source_dataset,
                "manip": manip,
                "feat_key": feat_key,
                "num_samples_for_stats": n,
            }
            torch.save(state, norm_params_savefile)
        else:
            state = torch.load(norm_params_savefile, map_location="cpu")
            self.mean = state["mean"]
            self.std = state["std"]

        self._mean_gpu = None
        self._std_gpu = None

    def _get_gpu_stats(self, device, dtype):
        if self._mean_gpu is None or self._mean_gpu.device != device:
            self._mean_gpu = self.mean.to(device=device, dtype=dtype, non_blocking=True)
            self._std_gpu = self.std.to(device=device, dtype=dtype, non_blocking=True)
        return self._mean_gpu, self._std_gpu

    def normalize_batch(self, batch_tensor):
        mean, std = self._get_gpu_stats(batch_tensor.device, batch_tensor.dtype)

        if batch_tensor.dim() == 4 and mean.dim() == 4:
            return (batch_tensor - mean) / std

        return (batch_tensor - mean.squeeze(0)) / std.squeeze(0)

    def denormalize(self, feat_tensor):
        was_numpy = isinstance(feat_tensor, np.ndarray)

        if was_numpy:
            feat_tensor = torch.from_numpy(
                np.asarray(feat_tensor)
            ).contiguous()

        single = False
        if feat_tensor.dim() == 3:
            feat_tensor = feat_tensor.unsqueeze(0)
            single = True

        mean, std = self._get_gpu_stats(
            feat_tensor.device,
            feat_tensor.dtype,
        )

        denorm = feat_tensor * std + mean

        if single:
            denorm = denorm.squeeze(0)

        return denorm
