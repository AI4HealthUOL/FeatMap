import os
import glob
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import re
import csv
import timm

"""Evaluation of the finetuned ConvNeXt and SwinV2 classifiers on the Stanford Cars image datasets."""


class AugmentedTestDataset(datasets.ImageFolder):
    def __init__(
        self, root, transform=None, keep_dirs=("direct", "qwen_gs_1_infsteps_10")
    ):
        super().__init__(root, transform=transform)
        self.manip_pattern = re.compile(r"^[^_]+_(.+)\.jpg$", re.IGNORECASE)

        filtered = []
        for path, cls in self.samples:
            if "resized" in path:
                continue

            aug_dir = os.path.basename(os.path.dirname(path))
            if any(k in aug_dir for k in keep_dirs):
                filtered.append((path, cls))

        self.samples = filtered

    def __getitem__(self, index):
        path, true_class_idx = self.samples[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)

        fname = os.path.basename(path)
        m = self.manip_pattern.match(fname)
        manip = m.group(1) if m else "unknown"

        return img, true_class_idx, manip


model_path = os.path.expandvars("$WORK/models/")
dataset_path = os.path.expandvars("$WORK/datasets/")
eval_path = os.path.expandvars("$WORK/evals/")
test_dir = os.path.join(dataset_path, "STANFORD_CARS/images/test")
test_augmented_dir = os.path.join(
    dataset_path, "STANFORD_CARS/images/augmented_test/")


def get_model_name(ckpt_name: str):
    if "convnext" in ckpt_name:
        return "convnext_base.fb_in22k_ft_in1k", 288
    elif "swinv2" in ckpt_name:
        return "swinv2_base_window12to24_192to384_22kft1k", 384


def evaluate(loader, with_manip=False, csv_path: str | None = None):
    total = correct1 = correct5 = 0
    per_manip = {} if with_manip else None
    results = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if with_manip:
                imgs, labels, manips = batch
            else:
                imgs, labels = batch
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits = model(imgs)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct1 += (preds == labels).sum().item()
            top5 = logits.topk(5, 1)[1]
            correct5 += (top5 == labels.unsqueeze(1)).sum().item()

            if with_manip:
                for m, p, l, t5 in zip(manips, preds.cpu(), labels.cpu(), top5.cpu()):
                    stats = per_manip.setdefault(
                        m, {"total": 0, "correct1": 0, "correct5": 0}
                    )
                    stats["total"] += 1
                    stats["correct1"] += int(p == l)
                    stats["correct5"] += int((t5 == l).any())

    if with_manip:
        for m, stats in sorted(per_manip.items()):
            t = stats["total"]
            top1 = stats["correct1"] / t
            top5 = stats["correct5"] / t
            results.append((m, t, top1, top5))
    else:
        results.append((correct1 / total, correct5 / total))

    if csv_path is not None:
        mode = "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                if with_manip:
                    writer.writerow(["manipulation", "total", "top1", "top5"])
                else:
                    writer.writerow(["top1", "top5"])
            for row in results:
                writer.writerow(row)


for ft_model_name in ["convnext_base.fb_in22k_ft_in1k_frozen_stages0-stages1-stages2-stages3_augmented_train_best.ckpt", "swinv2_base_window12to24_192to384_22kft1k_frozen__augmented_train_best.ckpt"]:

    base_model_name, img_size = get_model_name(ft_model_name)

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )

    original_test_ds = datasets.ImageFolder(root=test_dir, transform=transform)
    aug_test_ds = AugmentedTestDataset(test_augmented_dir, transform=transform)

    original_test_loader = DataLoader(
        original_test_ds, batch_size=128, shuffle=False, num_workers=7
    )
    aug_test_dl = DataLoader(aug_test_ds, batch_size=128,
                             shuffle=False, num_workers=7)

    ckpts = glob.glob(os.path.join(model_path, ft_model_name))
    if not ckpts:
        raise FileNotFoundError("no checkpoint found in " + model_path)
    ckpt_file = sorted(ckpts)[-1]

    model = timm.create_model(
        base_model_name,
        pretrained=False,
        num_classes=196,  # Stanford Cars
    )

    checkpoint = torch.load(ckpt_file, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k,
                  v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict, strict=True)

    model.eval().cuda()

    if (
        ft_model_name
        == "convnext_base.fb_in22k_ft_in1k_frozen_stages0-stages1-stages2-stages3_augmented_train_best.ckpt"
    ):
        evaluate(
            original_test_loader,
            csv_path=os.path.join(
                eval_path, "class_probs/original_test_aug_ft_results.csv"
            ),
        )

        evaluate(
            aug_test_dl,
            with_manip=True,
            csv_path=os.path.join(
                eval_path, "class_probs/augmented_test_aug_ft_results.csv"
            ),
        )
    elif (
        ft_model_name
        == "convnext_base.fb_in22k_ft_in1k_frozen_stages0-stages1-stages2-stages3_train_best.ckpt"
    ):
        evaluate(
            original_test_loader,
            csv_path=os.path.join(
                eval_path, "class_probs/original_test_ft_results.csv"),
        )

        evaluate(
            aug_test_dl,
            with_manip=True,
            csv_path=os.path.join(
                eval_path, "class_probs/augmented_test_ft_results.csv"),
        )
    elif (
        ft_model_name
        == "swinv2_base_window12to24_192to384_22kft1k_frozen__augmented_train_best.ckpt"
    ):
        evaluate(
            original_test_loader,
            csv_path=os.path.join(
                eval_path, "class_probs/swin_original_test_ft_results.csv"),
        )

        evaluate(
            aug_test_dl,
            with_manip=True,
            csv_path=os.path.join(
                eval_path, "class_probs/swin_augmented_test_ft_results.csv"),
        )
