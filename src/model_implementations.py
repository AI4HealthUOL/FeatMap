import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from eval_helpers import cosine_similarity_loss, CombinedLoss
from utils import get_mapping_run_paths

"""Model implementations for learning the mappings from original to target features"""

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


class MappingModel(pl.LightningModule):
    def __init__(self, config):
        super(MappingModel, self).__init__()

        self.model_type = config["model_type"]
        self.manipulation = config.get("manipulation", None)
        self.apply_transform_flag = config.get("apply_transform", False)

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.spatial_size = config.get("spatial_size", 9)
        self.lr = config.get("learning_rate", 1e-3)

        self.mse_scale = config.get("mse_scale", 0.5)
        self.cossim_scale = config.get("cossim_scale", 0.5)
        self.loss_fn = config.get("loss_function", "MSE")

        self.feature_key = config.get("feature_key", "feat2")
        self.batch_size = config.get("batch_size", 32)
        self.scheduler = config.get("scheduler", "ReduceLROnPlateau")


        self.model = get_model(
            self.model_type,
            self.input_dim,
            self.output_dim,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=config.get("dropout", 0.2),
            spatial_size=self.spatial_size,
        )

        if self.loss_fn == "MSE":
            self.criterion = nn.MSELoss()
        elif self.loss_fn == "COS_SIM":
            self.criterion = cosine_similarity_loss
        elif self.loss_fn == "COMBINED":
            self.criterion = CombinedLoss(
                mse_scale=self.mse_scale,
                cossim_scale=self.cossim_scale
            )

    def forward(self, x):
        # Applies spatial transform of the feature vector positions
        if self.apply_transform_flag and self.manipulation is not None:
            x = apply_spatial_transform(x, self.manipulation)

        if self.model_type in ["transformer", "cnn"]:
            return self.model(x)
        else:
            B, C, H, W = x.shape
            # Applies mapping model to transform each feature vector itself
            # Flatten spatial positions
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            # Apply single shared model
            mapped_vectors = self.model(x_flat)
            # Reshape back to spatial grid
            output = mapped_vectors.reshape(B, H, W, -1).permute(0, 3, 1, 2)

            return output

    def training_step(self, batch, batch_idx):
        orig_features, target_features, _ = batch

        outputs = self(orig_features)

        feat_loss = self.criterion(outputs, target_features)

        self.log(
            "train_loss",
            feat_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return feat_loss

    def validation_step(self, batch, batch_idx):
        orig_features, target_features, _ = batch

        outputs = self(orig_features)

        val_loss = self.criterion(outputs, target_features)

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
            self.parameters(), lr=self.lr, weight_decay=1e-4, foreach=True
        )
        if self.scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=8, factor=0.5, min_lr=1e-6
            )
        elif self.scheduler == "warmup":
            scheduler = pl.tuner.LinearWarmupScheduler(
                optimizer, warmup_steps=1000, min_lr=self.lr * 0.1
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


MODEL_REGISTRY = {
    "linear": lambda in_d, out_d, **kw: nn.Linear(in_d, out_d),

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
    num_feature_vectors=None,
    apply_transform=False,
):
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

    base = get_mapping_run_paths(
        model_path,
        extractor_name,
        feature_subdir=feature_subdir,
        dataset=dataset,
        model_type=model_type,
        manipulation=target_manipulation,
        apply_transform=apply_transform, 
    )

    checkpoint_prefix = f"{model_type}_{feature_key}_normalized_{normalized}_mapping_model"
    print("\n", base, checkpoint_prefix, "\n")

    model_checkpoint_path = get_best_checkpoint(
        base, checkpoint_prefix
    )

    print(
        f"Loading mapping checkpoint: {model_checkpoint_path}"
    )

    if model_checkpoint_path is None:
        return None

    state = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    return model