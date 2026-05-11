import os
import timm
import yaml
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import torch.nn.functional as F
from typing import Optional
from sklearn.model_selection import train_test_split
import numpy as np

"""
Fine-tuning pipeline for Stanford Cars classification.

This script trains image classifiers using pretrained ConvNeXt or SwinV2 backbones
from the timm library, focusing on controlled fine-tuning experiments.

Key characteristics:
- Dataset: Stanford Cars (train/augmented_train + test splits)
- Models: ConvNeXt / SwinV2 pretrained backbones
- Training strategy: backbone frozen, classifier head trained only
- Optional use of augmented image sets (direct + Qwen-generated)
- Stratified train/validation split (95/5)
- Experiment sweeps over model variants and training configurations

Training objective:
- Standard cross-entropy classification loss with label smoothing
- Evaluation via Top-1 and Top-5 accuracy

"""


class FilteredImageFolder(datasets.ImageFolder):
    def __init__(
        self, root, transform=None, target_transform=None, allowed_subdirs=None
    ):
        self.allowed_subdirs = allowed_subdirs or [
            "direct", "qwen_gs1_infsteps_10"]
        super().__init__(root, transform, target_transform)

    def find_samples(self, directory: str):
        samples = []
        class_name = os.path.basename(directory)
        dir_path = os.path.join(self.root, class_name)
        for subdir in self.allowed_subdirs:
            subdir_path = os.path.join(dir_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for file in os.listdir(subdir_path):
                if self._is_valid_image(file):
                    samples.append(
                        (os.path.join(subdir_path, file), class_name))
        return samples

    def _is_valid_image(self, filename):
        if not any(
            filename.lower().endswith(ext)
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ):
            return False

        # not useful for car classifier, so removing
        exclude_patterns = ["Remove_the_car_from_the_image"]
        for pattern in exclude_patterns:
            if pattern.lower() in filename.lower():
                return False

        return True


class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        use_augmented_train: bool,
        batch_size: int = 64,
        num_workers: int = 7,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.use_augmented_train = use_augmented_train
        self.num_classes = None

        self.train_transforms = transforms.Compose(
            [
                transforms.RandAugment(num_ops=1, magnitude=9),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(288, scale=(0.08, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(288),
                transforms.CenterCrop(288),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        if self.use_augmented_train:
            full_dataset = FilteredImageFolder(
                self.train_dir, allowed_subdirs=[
                    "direct", "qwen_gs1_infsteps_10"]
            )
        else:
            full_dataset = datasets.ImageFolder(self.train_dir)
        self.num_classes = len(full_dataset.classes)

        print(
            f"Loaded {len(full_dataset)} filtered samples across {len(full_dataset.classes)} classes"
        )

        labels = np.array([s[1] for s in full_dataset.samples])
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)), test_size=0.05, stratify=labels, random_state=42
        )

        self.train_dataset = TransformDataset(
            full_dataset, train_idx, self.train_transforms
        )
        self.val_dataset = TransformDataset(
            full_dataset, val_idx, self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )


class TransformDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        x, y = self.dataset[sample_idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def get_model_transforms(model_name):
    if "convnext" in model_name:
        size = 288
    elif "swinv2" in model_name:
        size = 384

    train_tf = transforms.Compose([
        transforms.RandAugment(num_ops=1, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


class FTClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 5e-5,
        frozen_stages: Optional[list[str]] = None,
        label_smoothing: float = 0.1,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze only classification head
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True

        elif hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits, y, label_smoothing=self.hparams.label_smoothing)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc1 = (preds == y).float().mean()

        top5_preds = logits.topk(5, 1, True, True)[1]
        acc5 = (top5_preds == y.unsqueeze(1)).float().sum(1).mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc/top1", acc1, prog_bar=True, on_epoch=True)
        self.log("val_acc/top5", acc5, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


torch.set_float32_matmul_precision("high")

model_path = os.path.expandvars("$WORK/models/")
dataset_path = os.path.expandvars("$WORK/datasets/")
log_path = os.path.expandvars("$WORK/logs/")

model_versions = [
    # "convnext_base.fb_in22k_ft_in1k",
    "swinv2_base_window12to24_192to384_22kft1k",
]


def freeze_suffix(stages):
    clean = [s.replace(".", "") for s in sorted(stages)]
    return "frozen_" + "-".join(clean)


frozen_stages_variants_convnext = [
    ["stages.0", "stages.1", "stages.2", "stages.3"],
]

frozen_stages_variants_swinv2 = [
    []
]

for model_name in model_versions:
    if "convnext" in model_name:
        frozen_stages_variants = frozen_stages_variants_convnext
    elif "swinv2" in model_name:
        frozen_stages_variants = frozen_stages_variants_swinv2

    for use_augmented_train in [True]:
        if use_augmented_train:
            train_dir = os.path.join(
                dataset_path, "STANFORD_CARS/images/augmented_train")
        else:
            train_dir = os.path.join(
                dataset_path, "STANFORD_CARS/images/train")

        if "convnext" in model_name:
            size = 288
            base_bs = 128
        elif "swinv2" in model_name:
            size = 384
            base_bs = 128

        for frozen_stages in frozen_stages_variants:
            if len(frozen_stages) == 4 or "convnext" in model_name:
                lr = 1e-3
                num_epochs = 30
                batch_size = base_bs
                label_smoothing = 0.1
                drop_rate = 0.3
            elif "swinv2" in model_name:
                lr = 1e-3
                num_epochs = 20
                batch_size = base_bs
                label_smoothing = 0.1
                drop_rate = 0.1

            if use_augmented_train:
                num_epochs = max(15, num_epochs // 2)

            val_split = 0.05

            train_tf, val_tf = get_model_transforms(model_name)

            data_module = StanfordCarsDataModule(
                train_dir=train_dir,
                use_augmented_train=use_augmented_train,
                batch_size=batch_size,
                num_workers=7,
                val_split=val_split,
            )

            data_module.train_transforms = train_tf
            data_module.val_transforms = val_tf
            data_module.setup()

            model = FTClassifier(
                model_name=model_name,
                num_classes=data_module.num_classes,
                lr=lr,
                frozen_stages=frozen_stages,
                label_smoothing=label_smoothing,
                drop_rate=drop_rate,
            )

            logger = TensorBoardLogger(
                save_dir=os.path.join(log_path, f"ft_{model_name}_cars"),
                name=f"{model_name}_{freeze_suffix(frozen_stages)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            if use_augmented_train:
                fn = f"{model_name}_{freeze_suffix(frozen_stages)}_augmented_train_best"
            else:
                fn = f"{model_name}_{freeze_suffix(frozen_stages)}_train_best"

            callbacks = [
                ModelCheckpoint(
                    monitor="val_acc/top1",
                    dirpath=model_path,
                    filename=fn,
                    save_top_k=1,
                    mode="max",
                ),
                EarlyStopping(monitor="val_acc/top1", patience=7,
                              mode="max", verbose=True),
            ]

            trainer = pl.Trainer(
                max_epochs=num_epochs,
                accelerator="gpu",
                devices=1,
                logger=logger,
                enable_model_summary=False,
                enable_progress_bar=True,
                callbacks=callbacks,
                log_every_n_steps=100,
                deterministic=False,
            )

            trainer.fit(model, datamodule=data_module)
