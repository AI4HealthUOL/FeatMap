import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import top_k_accuracy_score
import torch
import pandas as pd
import glob
import yaml
from itertools import product

"""Evaluation of the classification probabilities extracted with test_classifier.py"""

def compute_entropy(probs):
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1)


def compute_jsd(p, q):
    # normalize distributions
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)

    m = 0.5 * (p + q)

    # KL divergence: KL(P || M) + KL(Q || M)
    # with log2 scales between 0-1
    kl_p_m = (p * (p / m).log2()).sum(dim=1)
    kl_q_m = (q * (q / m).log2()).sum(dim=1)

    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd


def compute_top1_prob_diff(probs_orig, probs_other):
    top1_orig = probs_orig.max(dim=1).values
    top1_other = probs_other.max(dim=1).values
    return (top1_orig - top1_other).abs()


def topk_acc(probs, targets, k=10):
    _, topk_idx = torch.topk(probs, min(k, probs.shape[1]), dim=1)
    hits = torch.any(topk_idx == targets.unsqueeze(1), dim=1)
    return hits.float().mean().item()


def top1_agreement(probs_a, probs_b):
    pred_a = probs_a.argmax(dim=1)
    pred_b = probs_b.argmax(dim=1)
    return (pred_a == pred_b).float().mean().item()


def compute_metrics(probs_orig, probs_edit, probs_mapped, targets):
    """
    Compute metrics from the classification output probabilities.
    
    Evaluates all original, edited and mapped classification probablities.
    """
    metrics = {}
    for k in [1, 5]:
        yield_acc = {
            f"acc_orig_top{k}": topk_acc(probs_orig, targets, k=k),
            f"acc_edit_top{k}": topk_acc(probs_edit, targets, k=k),
            f"acc_mapped_top{k}": topk_acc(probs_mapped, targets, k=k),
        }
        for key, value in yield_acc.items():
            metrics[key] = {"mean": float(value)}

    entropy_orig = compute_entropy(
        probs_orig) / torch.log(torch.tensor(probs_orig.shape[1], dtype=torch.float32))
    entropy_edit = compute_entropy(
        probs_edit) / torch.log(torch.tensor(probs_edit.shape[1], dtype=torch.float32))
    entropy_mapped = compute_entropy(
        probs_mapped) / torch.log(torch.tensor(probs_mapped.shape[1], dtype=torch.float32))

    jsd_orig_edit = compute_jsd(probs_orig, probs_edit)
    jsd_orig_mapped = compute_jsd(probs_orig, probs_mapped)

    top1_diff_edit = compute_top1_prob_diff(probs_orig, probs_edit)
    top1_diff_mapped = compute_top1_prob_diff(probs_orig, probs_mapped)

    top1_agree_orig_mapped = top1_agreement(probs_orig, probs_mapped)
    top1_agree_edit_mapped = top1_agreement(probs_edit, probs_mapped)

    def mean_std(values):
        return {"mean": values.mean().item(), "std": values.std().item()}

    metrics.update(
        {
            "entropy_orig": mean_std(entropy_orig),
            "entropy_edit": mean_std(entropy_edit),
            "entropy_mapped": mean_std(entropy_mapped),
            "jsd_orig_edit": mean_std(jsd_orig_edit),
            "jsd_orig_mapped": mean_std(jsd_orig_mapped),
            "top1_prob_diff_edit": mean_std(top1_diff_edit),
            "top1_prob_diff_mapped": mean_std(top1_diff_mapped),
            "top1_agreement_orig_mapped": {"mean": top1_agree_orig_mapped},
            "top1_agreement_edit_mapped": {"mean": top1_agree_edit_mapped},
            "n_samples": len(targets),
        }
    )

    return metrics


evals_path = os.path.expandvars("$WORK/evals_mapping/convnext")
feat_keys = ["feat0", "feat1", "feat2", "feat3"]
dataset = "STANFORD_CARS"
model_types = ["linear", "linear_rotation_90", "linear_rotation_180",
               "linear_rotation_270", "linear_mirror_h", "linear_mirror_v", "mlp", "cnn", "transformer"]


def normalize_model_name(model_type):
    if model_type.startswith("linear"):
        return "linear"
    return model_type


def build_prob_path(base_dir, prefix, dataset, manip, feature_key, model_type=None):
    model_part = "" if model_type is None else f"_{model_type}"
    return os.path.join(
        base_dir,
        f"{prefix}_{dataset}_{manip}_{feature_key}{model_part}.pt"
    )


def flatten_metrics(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[f"{k}_{sk}"] = sv
        else:
            out[k] = v
    return out


def main():
    evals_path = "evals/convnext"
    dataset = "STANFORD_CARS"

    with open("../../../config/test_classifier.yaml") as f:
        config = yaml.safe_load(f)

    all_results = []

    # Evaluate for which finetuned model?
    for ft_type in ["augmented_head"]:
        # Probabilities were extracted with test_classifier.py
        prob_dir = os.path.join(evals_path, f"class_probs/{ft_type}")

        # Goes over each manipulation group and calculates the metrics per sample
        for dataset_name, groups in config["target_manipulations"].items():
            for group_name, group_cfg in groups.items():
                manipulations = group_cfg["manipulations"]
                allowed_models = group_cfg["models"]

                for manip, feature_key in product(manipulations, config["feature_keys"].keys()):

                    for model_type in allowed_models + [""]:

                        norm_model = normalize_model_name(model_type)

                        orig_path = build_prob_path(
                            prob_dir, "probs_orig", dataset, manip, feature_key)
                        edit_path = build_prob_path(
                            prob_dir, "probs_edit", dataset, manip, feature_key)
                        mapped_path = build_prob_path(
                            prob_dir, "probs_mapped", dataset, manip, feature_key, model_type)

                        if not (os.path.exists(orig_path) and os.path.exists(edit_path) and os.path.exists(mapped_path)):
                            continue

                        data_orig = torch.load(orig_path, map_location="cpu")
                        data_edit = torch.load(edit_path, map_location="cpu")
                        data_mapped = torch.load(
                            mapped_path, map_location="cpu")

                        probs_orig = data_orig["probs"]
                        probs_edit = data_edit["probs"]
                        probs_mapped = data_mapped["probs"]
                        targets = data_orig["targets"]

                        n = min(len(probs_orig), len(
                            probs_edit), len(probs_mapped))

                        probs_orig = probs_orig[:n]
                        probs_edit = probs_edit[:n]
                        probs_mapped = probs_mapped[:n]
                        targets = targets[:n]

                        if not (len(probs_orig) == len(probs_edit) == len(probs_mapped)):
                            continue

                        metrics = compute_metrics(
                            probs_orig,
                            probs_edit,
                            probs_mapped,
                            targets
                        )

                        flat = flatten_metrics(metrics)

                        flat.update({
                            "model_type": norm_model,
                            "manipulation": manip,
                            "feature_key": feature_key,
                            "dataset": dataset
                        })

                        all_results.append(flat)

        df_all = pd.DataFrame(all_results)

        metrics_dir = os.path.join(
            evals_path, f"class_probs/metrics/{ft_type}")
        os.makedirs(metrics_dir, exist_ok=True)

        csv_path = os.path.join(metrics_dir, f"metrics_{dataset}_all.csv")
        df_all.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
