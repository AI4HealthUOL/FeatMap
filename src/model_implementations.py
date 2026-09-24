import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from eval_helpers import LossModule
from utils import get_mapping_run_paths
import time

"""Model implementations for learning the mappings from original to target features"""


class OrthogonalProcrustesMapping(nn.Module):
    """
    Closed-form orthogonal linear map:
        y = x @ R,  R ∈ O(d_out) (or rectangular orthogonal if dims differ)

    Assumes input/output are flattened feature vectors: [N, C].
    For spatial features [N, C, H, W], flatten H,W before calling fit().
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.register_buffer("R", torch.zeros(input_dim, output_dim))
        self._fitted = False

    @torch.no_grad()
    def fit(self, X, Y):
        """
        X: [N, input_dim]
        Y: [N, output_dim]

        Solves: min_R ||X @ R - Y||_F  s.t.  R^T R = I
        """
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D: [N, dim]")

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")

        if X.shape[1] != self.input_dim or Y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected X: [*, {self.input_dim}], Y: [*, {self.output_dim}], "
                f"got {X.shape} and {Y.shape}"
            )

        # Center
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Cross-covariance
        C = X.T @ Y  # [input_dim, output_dim]

        # SVD
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)

        # Orthogonal map
        R = U @ Vh  # [input_dim, output_dim]

        self.R.copy_(R)
        self._fitted = True

    def forward(self, x):
        if not self._fitted:
            raise RuntimeError("OrthogonalProcrustesMapping must be fitted before forward()")

        if x.ndim == 2:
            return x @ self.R

        # Handle [N, C, H, W]
        N, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        y_flat = x_flat @ self.R
        y = y_flat.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        return y

class RidgeRegressionMapping(nn.Module):
    """
    Affine ridge regression:

        y_hat = (x - x_mean) @ W + y_mean

    W = (Xc.T @ Xc + gamma I)^(-1) @ Xc.T @ Yc
    """

    def __init__(self, input_dim, output_dim, gamma=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = float(gamma)

        self.register_buffer(
            "W",
            torch.zeros(input_dim, output_dim),
        )
        self.register_buffer(
            "x_mean",
            torch.zeros(input_dim),
        )
        self.register_buffer(
            "y_mean",
            torch.zeros(output_dim),
        )
        self.register_buffer(
            "fitted",
            torch.tensor(False, dtype=torch.bool),
        )

    @torch.no_grad()
    def fit(self, X, Y):
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D: [N, dim]")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "X and Y must have the same number of samples"
            )

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {X.shape[1]}"
            )

        if Y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected output dimension {self.output_dim}, "
                f"got {Y.shape[1]}"
            )

        X = X.to(device=self.W.device, dtype=self.W.dtype)
        Y = Y.to(device=self.W.device, dtype=self.W.dtype)

        self.x_mean.copy_(X.mean(dim=0))
        self.y_mean.copy_(Y.mean(dim=0))

        Xc = X - self.x_mean
        Yc = Y - self.y_mean

        XtX = Xc.T @ Xc
        XtY = Xc.T @ Yc

        eye = torch.eye(
            self.input_dim,
            device=X.device,
            dtype=X.dtype,
        )

        W = torch.linalg.solve(
            XtX + self.gamma * eye,
            XtY,
        )

        self.W.copy_(W)
        self.fitted.fill_(True)

    def forward(self, x):
        if not bool(self.fitted):
            raise RuntimeError(
                "RidgeRegressionMapping must be fitted before forward()"
            )

        if x.ndim == 2:
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected last dimension {self.input_dim}, "
                    f"got {x.shape[-1]}"
                )

            return (x - self.x_mean) @ self.W + self.y_mean

        if x.ndim == 4:
            B, C, H, W = x.shape

            if C != self.input_dim:
                raise ValueError(
                    f"Expected {self.input_dim} input channels, "
                    f"got {C}"
                )

            x_flat = (
                x.permute(0, 2, 3, 1)
                .reshape(-1, C)
            )

            y_flat = (
                (x_flat - self.x_mean) @ self.W
                + self.y_mean
            )

            return (
                y_flat
                .reshape(B, H, W, self.output_dim)
                .permute(0, 3, 1, 2)
            )

        raise ValueError(
            f"Expected 2D or 4D input, got {x.ndim}D"
        )

class NonLinearMapping(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.2):
        super(NonLinearMapping, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class CNNMapping(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=512, kernel_size=3, dropout=0.2
    ):
        super(CNNMapping, self).__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, output_dim,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerMapping(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers=4,
        num_heads=8,
        hidden_dim=512,
        dropout=0.1,
        spatial_size=9,
    ):
        super().__init__()
        self.spatial_size = spatial_size

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.pre_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x

class AdditiveMapping(nn.Module):
    """
    y_hat = x + delta

    Requires input_dim == output_dim.
    """
    def __init__(self, input_dim, output_dim, init_delta=None):
        super().__init__()

        if input_dim != output_dim:
            raise ValueError(
                "AdditiveMapping requires input_dim == output_dim. "
                f"Got {input_dim} and {output_dim}."
            )

        if init_delta is None:
            init_delta = torch.zeros(output_dim)

        self.delta = nn.Parameter(init_delta.clone().float())

    def forward(self, x):
        # x: [N, C, H, W] or [N, C]
        shape = [1, -1] + [1] * (x.ndim - 2)
        return x + self.delta.view(*shape)

class LocalNonsharedLinearMapping(nn.Module):
    """
    Position-specific linear map:
        y[b, :, h, w] = W[h, w] @ x[b, :, h, w] + b[h, w]

    weight shape: [H, W, output_dim, input_dim]
    bias shape:   [H, W, output_dim]
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        spatial_size,
        bias=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_size = spatial_size

        self.weight = nn.Parameter(
            torch.empty(
                spatial_size,
                spatial_size,
                output_dim,
                input_dim,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    spatial_size,
                    spatial_size,
                    output_dim,
                )
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.spatial_size):
            for w in range(self.spatial_size):
                nn.init.xavier_uniform_(self.weight[h, w])

    def forward(self, x):
        # x: [B, C_in, H, W]
        B, C, H, W = x.shape

        if H != self.spatial_size or W != self.spatial_size:
            raise ValueError(
                f"Expected spatial size "
                f"{self.spatial_size}x{self.spatial_size}, "
                f"got {H}x{W}."
            )

        # [B, H, W, C]
        x_hw = x.permute(0, 2, 3, 1)

        # [B, H, W, output_dim]
        y = torch.einsum(
            "bhwi,hwoi->bhwo",
            x_hw,
            self.weight,
        )

        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)

        return y.permute(0, 3, 1, 2)

class MappingModel(pl.LightningModule):

    def __init__(self, config):
        super(MappingModel, self).__init__()

        self.model_type = config["model_type"]

        self.manipulation = config.get("manipulation", None)

        self.apply_transform_flag = config.get(
            "apply_transform",
            False,
        )

        self.input_dim = config["input_dim"]

        self.output_dim = config["output_dim"]

        self.spatial_size = config.get("spatial_size", 9)

        self.lr = config.get("learning_rate", 1e-3)

        self.feature_key = config.get("feature_key", "feat2")

        self.batch_size = config.get("batch_size", 32)

        self.scheduler = config.get(
            "scheduler",
            "ReduceLROnPlateau",
        )

        self.loss_name = config.get(
            "loss_function",
            "mse",
        )

        self.loss_params = config.get(
            "loss_params",
            {},
        )

        self.model = get_model(
            self.model_type,
            self.input_dim,
            self.output_dim,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=config.get("dropout", 0.2),
            spatial_size=self.spatial_size,
            num_layers=config.get("num_layers", 4),
            num_heads=config.get("num_heads", 8),
            kernel_size=config.get("kernel_size", 3),
        )


        self.criterion = LossModule.create(
            self.loss_name,
            self.loss_params,
        )

    def forward(self, x):
        if (
            self.apply_transform_flag
            and self.manipulation is not None
        ):
            x = apply_spatial_transform(
                x,
                self.manipulation,
            )

        if self.model_type in [
                "transformer",
                "cnn",
                "additive",
                "local_nonshared_linear",
            ]:
            return self.model(x)

        B, C, H, W = x.shape

        x_flat = (
            x.permute(0, 2, 3, 1)
            .reshape(-1, C)
        )

        mapped_vectors = self.model(x_flat)

        output = (
            mapped_vectors
            .reshape(B, H, W, -1)
            .permute(0, 3, 1, 2)
        )

        return output

    def training_step(self, batch, batch_idx):
        orig_features, target_features, _ = batch

        outputs = self(orig_features)

        loss = self.criterion(
            outputs,
            target_features,
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        orig_features, target_features, _ = batch

        outputs = self(orig_features)

        val_loss = self.criterion(
            outputs,
            target_features,
        )

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return val_loss

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            foreach=True,
        )

        if self.scheduler == "ReduceLROnPlateau":

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=8,
                factor=0.5,
                min_lr=1e-6,
            )

        elif self.scheduler == "warmup":

            scheduler = pl.tuner.LinearWarmupScheduler(
                optimizer,
                warmup_steps=1000,
                min_lr=self.lr * 0.1,
            )

        else:
            scheduler = None

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


MODEL_REGISTRY = {
    "linear": lambda in_d, out_d, **kw: nn.Linear(in_d, out_d),

    "additive": lambda in_d, out_d, **kw: AdditiveMapping(
        in_d,
        out_d,
        init_delta=kw.get("init_delta", None),
    ),
    
    "local_nonshared_linear": lambda in_d, out_d, **kw: LocalNonsharedLinearMapping(
        in_d,
        out_d,
        spatial_size=kw.get("spatial_size", 9),
        bias=kw.get("bias", True),
    ),

    "mlp": lambda in_d, out_d, **kw: NonLinearMapping(
        in_d, out_d,
        hidden_dim=kw.get("hidden_dim", 512),
        dropout=kw.get("dropout", 0.2),
    ),

    "cnn": lambda in_d, out_d, **kw: CNNMapping(
        in_d, out_d,
        hidden_dim=kw.get("hidden_dim", 512),
        kernel_size=kw.get("kernel_size", 3),
        dropout=kw.get("dropout", 0.2),
    ),

    "transformer": lambda in_d, out_d, **kw: TransformerMapping(
        in_d, out_d,
        num_layers=kw.get("num_layers", 4),
        num_heads=kw.get("num_heads", 8),
        hidden_dim=kw.get("hidden_dim", 512),
        dropout=kw.get("dropout", 0.1),
        spatial_size=kw.get("spatial_size", 9),
    ),

    # Closed-form baselines
    "orthogonal_procrustes": lambda in_d, out_d, **kw: OrthogonalProcrustesMapping(in_d, out_d),
    "ridge": lambda in_d, out_d, **kw: RidgeRegressionMapping(
        in_d, out_d, gamma=kw.get("gamma", 1.0)
    ),
}


def get_model(model_type, input_dim, output_dim, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}")

    return MODEL_REGISTRY[model_type](input_dim, output_dim, **kwargs)

def apply_spatial_transform(x, manipulation):
    if manipulation == "rotation_90":
        x = torch.rot90(x, k=1, dims=[2, 3])
    elif manipulation == "rotation_180":
        x = torch.rot90(x, k=2, dims=[2, 3])
    elif manipulation == "rotation_270":
        x = torch.rot90(x, k=3, dims=[2, 3])
    elif manipulation == "mirror_h":
        x = torch.flip(x, dims=[3])
    elif manipulation == "mirror_v":
        x = torch.flip(x, dims=[2])

    return x


def get_best_checkpoint(checkpoint_dir, checkpoint_prefix):
    """
    Retrieves the latest/best checkpoint file matching a given prefix.

    Assumes checkpoint filenames follow PyTorch Lightning conventions:
        <prefix>-vX.ckpt

    Where v is the version id.

    Args:
        checkpoint_dir (str): Directory containing checkpoints.
        checkpoint_prefix (str): Prefix used during training.

    Returns:
        str or None: Path to selected checkpoint.
    """
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
import os
import re
import glob
import torch

from utils import get_mapping_run_paths


def get_best_checkpoint_new(checkpoint_dir: str, checkpoint_prefix: str):
    try:
        checkpoint_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith(checkpoint_prefix) and f.endswith(".ckpt")
        ]

        def version_key(fname: str):
            m = re.search(r"-v(\d+)", fname)
            return int(m.group(1)) if m else 0

        checkpoint_files.sort(key=version_key, reverse=True)

        if not checkpoint_files:
            return None

        return os.path.join(checkpoint_dir, checkpoint_files[0])
    except FileNotFoundError:
        return None


import os
import glob
import torch


def make_safe_name(value):
    """
    Keep this consistent with the training code.

    For your current names, this leaves values such as:
        mse_median_cos
        grayscale
        rotation_90
    unchanged.
    """
    value = str(value)
    value = value.strip()
    value = value.replace("/", "_")
    value = value.replace("\\", "_")
    value = value.replace(" ", "_")
    return value


def get_best_checkpoint_new(directory, prefix):
    """
    Return the checkpoint beginning with prefix.

    Supports:
        prefix.ckpt
        prefix-*.ckpt
        prefix*.ckpt

    If multiple files exist, prefers files containing 'best'.
    """
    if not os.path.isdir(directory):
        print(f"Checkpoint directory does not exist: {directory}")
        return None

    candidates = sorted(
        glob.glob(os.path.join(directory, f"{prefix}*.ckpt"))
    )

    if not candidates:
        print(
            f"No checkpoint found.\n"
            f"  directory: {directory}\n"
            f"  prefix:    {prefix}"
        )
        return None

    if len(candidates) == 1:
        return candidates[0]

    best_candidates = [
        path for path in candidates
        if "best" in os.path.basename(path).lower()
    ]

    if len(best_candidates) == 1:
        return best_candidates[0]

    print(
        f"Multiple checkpoints found for prefix '{prefix}'. "
        f"Using the first one:\n"
        + "\n".join(candidates)
    )

    return candidates[0]


def get_checkpoint_directory(
    model_path,
    extractor_name,
    feature_subdir,
    dataset,
    model_type,
    manipulation,
    feature_key,
    loss_name,
    apply_flag
):
    if apply_flag:
        m = "applied"
    else:
        m = "learned"

    base = os.path.join(
        os.path.expandvars(model_path),
        extractor_name,
        feature_subdir,
        dataset,
        m,
        model_type,
        manipulation,
        feature_key,
        f"loss-{make_safe_name(loss_name)}",
    )

    # Backward-compat fix: if path contains "<extractor>/<extractor>/",
    # collapse the duplicate segment when the directory doesn't exist.
    # This handles cases where feature_subdir was accidentally set to extractor_name.
    if not os.path.isdir(base):
        dup_pattern = os.path.join(extractor_name, extractor_name)
        if dup_pattern.replace("\\", "/") in base.replace("\\", "/"):
            candidate = base.replace(dup_pattern, extractor_name, 1)
            if os.path.isdir(candidate):
                return candidate

    return base


def resolve_checkpoint(
    model_path,
    extractor_name,
    feature_subdir,
    dataset,
    model_type,
    manipulation,
    feature_key,
    loss_name,
    normalized,
    apply_flag
):
    safe_loss = make_safe_name(loss_name)
    safe_manipulation = make_safe_name(manipulation)

    checkpoint_dir = get_checkpoint_directory(
        model_path=model_path,
        extractor_name=extractor_name,
        feature_subdir=feature_subdir,
        dataset=dataset,
        model_type=model_type,
        manipulation=manipulation,
        feature_key=feature_key,
        loss_name=loss_name,
        apply_flag=apply_flag
    )

    # New layout
    new_prefix = (
        f"{extractor_name}_"
        f"{model_type}_"
        f"{feature_key}_"
        f"{safe_manipulation}_"
        f"loss-{safe_loss}_"
        f"normalized-{str(bool(normalized))}_"
        "mapping_model"
    )

    checkpoint_path = get_best_checkpoint_new(
        checkpoint_dir,
        new_prefix,
    )

    if checkpoint_path is not None:
        return checkpoint_path

    # Alternative: normalized_True vs normalized-True
    alternative_prefix = (
        f"{extractor_name}_"
        f"{model_type}_"
        f"{feature_key}_"
        f"{safe_manipulation}_"
        f"loss-{safe_loss}_"
        f"normalized_{str(bool(normalized))}_"
        "mapping_model"
    )

    checkpoint_path = get_best_checkpoint_new(
        checkpoint_dir,
        alternative_prefix,
    )

    if checkpoint_path is not None:
        return checkpoint_path

    # Fallback: single checkpoint in directory
    all_checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    )

    if len(all_checkpoints) == 1:
        only_checkpoint = all_checkpoints[0]
        print(
            "Using the only checkpoint in the expected directory:\n"
            f"  {only_checkpoint}"
        )
        return only_checkpoint

    print(
        "Could not resolve checkpoint:\n"
        f"  directory: {checkpoint_dir}\n"
        f"  expected prefix: {new_prefix}\n"
        f"  alternative prefix: {alternative_prefix}"
    )

    return None

from collections import OrderedDict
def normalize_state_dict_to_model(state_dict, model):
    """
    Adapt common checkpoint formats to the exact keys expected by `model`.
    Only accepts mappings where every checkpoint tensor can be matched
    unambiguously to a current model key.
    """
    target_keys = set(model.state_dict().keys())
    source_keys = list(state_dict.keys())

    # Remove non-tensor entries, if any.
    source_state = {
        k: v for k, v in state_dict.items()
        if torch.is_tensor(v)
    }

    # Candidate transformations, ordered from most specific to broadest.
    transforms = [
        lambda k: k,
        lambda k: f"model.{k}",
        lambda k: f"model.net.{k}",
        lambda k: k.removeprefix("model."),
        lambda k: k.removeprefix("module."),
        lambda k: k.removeprefix("model.module."),
        lambda k: k.removeprefix("net."),
        lambda k: f"model.{k.removeprefix('module.')}",
        lambda k: f"model.{k.removeprefix('net.')}",
        lambda k: f"model.net.{k.removeprefix('net.')}",
    ]

    # First try transformations globally.
    for transform in transforms:
        candidate = {
            transform(k): v
            for k, v in source_state.items()
        }

        if set(candidate).issubset(target_keys):
            return OrderedDict(candidate)

    # Handle direct Sequential checkpoints:
    # 0.weight -> model.net.0.weight
    direct_sequential = {
        f"model.net.{k}": v
        for k, v in source_state.items()
    }

    if set(direct_sequential).issubset(target_keys):
        return OrderedDict(direct_sequential)

    # Handle direct Linear checkpoints:
    # weight -> model.weight
    direct_model = {
        f"model.{k}": v
        for k, v in source_state.items()
    }

    if set(direct_model).issubset(target_keys):
        return OrderedDict(direct_model)

    # Handle checkpoints that contain an extra wrapper prefix by suffix matching.
    remapped = OrderedDict()
    used_target_keys = set()

    for source_key, value in source_state.items():
        matches = [
            target_key
            for target_key in target_keys
            if target_key == source_key
            or target_key.endswith("." + source_key)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely map checkpoint key "
                f"{source_key!r} to current model keys. "
                f"Candidates: {matches}"
            )

        target_key = matches[0]

        if target_key in used_target_keys:
            raise RuntimeError(
                f"Multiple checkpoint keys map to {target_key!r}"
            )

        remapped[target_key] = value
        used_target_keys.add(target_key)

    return remapped

def load_model(
    dataset,
    extractor_name,
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
    loss_name,
    num_feature_vectors=None,
    apply_transform=False,
    allow_convnext_legacy=True,
):
    spatial_size = (
        int(num_feature_vectors ** 0.5)
        if num_feature_vectors
        else 9
    )

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

    checkpoint_path = resolve_checkpoint(
        model_path=model_path,
        extractor_name=extractor_name,
        feature_subdir=feature_subdir,
        dataset=dataset,
        model_type=model_type,
        manipulation=target_manipulation,
        feature_key=feature_key,
        loss_name=loss_name,
        normalized=normalized,
        apply_flag=apply_transform
    )

    if checkpoint_path is None:
        return None

    print(f"Loading mapping checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint type: "
            f"{type(checkpoint)}"
        )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint and isinstance(
        checkpoint["model"],
        dict,
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Normalize keys to match current MappingModel structure
    state_dict = normalize_state_dict_to_model(
        state_dict,
        model,
    )

    if model_type == "ridge":
        model.load_state_dict(state_dict, strict=True)
    else:
        incompatible = model.load_state_dict(
            state_dict,
            strict=False,
        )

    if incompatible.missing_keys:
        print(
            "Missing checkpoint keys (may be harmless if due to architecture changes):"
            f" {incompatible.missing_keys}"
        )

    if incompatible.unexpected_keys:
        print(
            "Unexpected checkpoint keys (may be harmless if due to architecture changes):"
            f" {incompatible.unexpected_keys}"
        )

    # --- Restore _fitted for closed-form models ---
    is_closed_form = model_type in ("orthogonal_procrustes", "ridge")
    was_closed_form_checkpoint = checkpoint.get("closed_form", False)

    if is_closed_form and was_closed_form_checkpoint:
        if model_type == "ridge":
            if not bool(model.model.fitted):
                raise RuntimeError(
                    "Ridge checkpoint was marked closed_form but its "
                    "fitted buffer is False."
                )

        elif model_type == "orthogonal_procrustes":
            model.model._fitted = True

    model.eval()

    # Useful for CSV reporting.
    model.checkpoint_path = os.path.abspath(
        checkpoint_path
    )

    return model