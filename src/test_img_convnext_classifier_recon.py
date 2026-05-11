import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv
import timm
import json

torch.backends.cudnn.benchmark = True

"""Evaluation of the finetuned ConvNeXt classifier on the Stanford Cars image datasets."""


class MappingDataset(Dataset):
    def __init__(self, img_paths, id_to_target, transform=None):
        self.transform = transform
        self.samples = []

        for path in img_paths:
            fname = os.path.basename(path)
            img_id = fname.split("_")[0]

            if img_id not in id_to_target:
                print("not in id_to_target")
                continue

            label = id_to_target[img_id]

            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def evaluate(loader):

    total = correct1 = correct5 = 0

    with torch.no_grad():
        for imgs, labels in loader:

            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits = model(imgs)

            preds = logits.argmax(dim=1)
            total += labels.size(0)

            correct1 += (preds == labels).sum().item()

            top5 = logits.topk(5, dim=1).indices
            correct5 += (top5 == labels.view(-1, 1)).any(dim=1).sum().item()

    return correct1 / total, correct5 / total


model_path = os.path.expandvars("models/")
dataset_path = os.path.expandvars("datasets/")
eval_path = os.path.expandvars("evals_mapping/")

mapping_models = ["transformer", "linear", "mlp", "cnn"]

transform = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)

root_dir = os.path.join(dataset_path, "STANFORD_CARS")

metadata_path = os.path.join(
    root_dir,
    "features/augmented_test/direct",
    "STANFORD_CARS_test_direct_metadata.json",
)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

id_to_target = {
    entry["original_id"]: entry["target"]
    for entry in metadata.values()
}

for classifier_name in ["convnext_base.fb_in22k_ft_in1k_frozen_stages0-stages1-stages2-stages3_augmented_train_best.ckpt"]:
    ckpt_file = os.path.join(
        model_path,
        classifier_name,
    )

    model = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k",
        pretrained=True,
        num_classes=196,
    )

    checkpoint = torch.load(ckpt_file, map_location="cpu")

    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
    }

    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.cuda()

    results = []

    for mapping_model in mapping_models:

        model_root = os.path.join(eval_path, mapping_model)

        if not os.path.exists(model_root):
            continue

        for mapping_type in os.listdir(model_root):

            base = os.path.join(model_root, mapping_type, "STANFORD_CARS")
            if not os.path.exists(base):
                continue

            for feat in os.listdir(base):

                feat_dir = os.path.join(base, feat)

                for manip in os.listdir(feat_dir):

                    manip_dir = os.path.join(feat_dir, manip)

                    mapped_dir = os.path.join(manip_dir, "mapped_images")
                    recon_dir = os.path.join(manip_dir, "reconstructed_images")

                    mapped_paths = glob.glob(os.path.join(
                        mapped_dir, "*.png")) if os.path.exists(mapped_dir) else []
                    recon_paths = glob.glob(os.path.join(
                        recon_dir, "*.png")) if os.path.exists(recon_dir) else []

                    if len(mapped_paths) == 0 and len(recon_paths) == 0:
                        continue

                    print(mapping_model, mapping_type, feat, manip)

                    if mapped_paths:

                        dataset = MappingDataset(
                            mapped_paths, id_to_target, transform
                        )

                        loader = DataLoader(
                            dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True
                        )

                        top1, top5 = evaluate(loader)

                        results.append([
                            mapping_model,
                            mapping_type,
                            feat,
                            manip,
                            "mapped",
                            top1,
                            top5,
                        ])

                    if recon_paths:
                        dataset = MappingDataset(
                            recon_paths, id_to_target, transform
                        )

                        loader = DataLoader(
                            dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True
                        )

                        top1, top5 = evaluate(loader)

                        results.append([
                            mapping_model,
                            mapping_type,
                            feat,
                            manip,
                            "edit",
                            top1,
                            top5,
                        ])

    os.makedirs(os.path.join(eval_path, "class_probs"), exist_ok=True)

    if classifier_name == "convnext_base.fb_in22k_ft_in1k_frozen_stages0-stages1-stages2-stages3_augmented_train_best.ckpt":
        results_name = "class_probs/recon_augmented_ft_test_results.csv"
    else:
        results_name = "class_probs/recon_ft_test_results.csv"

    with open(os.path.join(eval_path, results_name), "w") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "mapping_type",
            "feature",
            "manipulation",
            "image_type",
            "top1",
            "top5",
        ])

        writer.writerows(results)
