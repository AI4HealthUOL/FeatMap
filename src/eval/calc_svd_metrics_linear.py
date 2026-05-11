import os
import numpy as np
import pandas as pd

def compute_metrics(svd_results, rank_threshold=1e-8):
    """Compute metrics from SVD singular values to characterize linear mapping matrices."""
    metrics = {}
    sv_list = []

    for key, svd in svd_results.items():
        S = np.asarray(svd["S"], dtype=np.float64)

        eps = 1e-12
        total = np.sum(S**2) + eps
        p = (S**2) / total

        effective_rank = np.exp(-np.sum(p * np.log(p + eps)))
        spectral_norm = float(S[0])
        frob_norm = float(np.sqrt(total))
        entropy = float(-np.sum(p * np.log(p + eps)))
        energy_concentration = float((S[0]**2) / total)

        S_norm = S / (np.mean(S) + eps)

        metrics[key] = {
            "rank": int(np.sum(S > rank_threshold)),
            "spectral_norm": spectral_norm,
            "frobenius_norm": frob_norm,
            "rmse_from_one": float(np.sqrt(np.mean((S_norm - 1) ** 2))),
            "near_one_pct": float(np.mean(np.abs(S_norm - 1) < 0.1) * 100),
            "effective_rank": float(effective_rank),
            "spectral_entropy": entropy,
            "energy_concentration": energy_concentration,
            "top1_var": float((S[0]**2) / total),
            "top5_var": float(np.sum(S[:5]**2) / total),
            "top10_var": float(np.sum(S[:10]**2) / total),
        }

        for i, s in enumerate(S, 1):
            sv_list.append({
                "manipulation": key,
                "rank": i,
                "singular_value": float(s),
                "singular_value_sq": float(s**2),
            })

    return metrics, sv_list


def load_svd_results(path):
    svd_results = {}

    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            svd_results[key] = data[key].item()

    return svd_results


model_path = os.path.expandvars("models/")
evals_path = os.path.expandvars("evals/")
output_dir = os.path.join(model_path, "svd_analysis")
os.makedirs(evals_path, exist_ok=True)

backbone_feats = {
    "convnext": ["feat0", "feat1", "feat2", "feat3"],
    "swinv2": ["feat0"],
}

all_svs = []

dataset = "STANFORD_CARS"

for backbone, feat_keys in backbone_feats.items():
    for feat_key in feat_keys:

        svd_file = os.path.join(
            output_dir,
            f"svd_{dataset}_{backbone}_{feat_key}.npz"
        )

        print("Loading:", svd_file)

        if not os.path.exists(svd_file):
            print("Missing:", svd_file)
            continue

        svd_results = load_svd_results(svd_file)
        basic_metrics, sv_list = compute_metrics(svd_results)

        for x in sv_list:
            x["feat_key"] = feat_key
            x["backbone"] = backbone

        all_svs.extend(sv_list)

        df = pd.DataFrame([
            {
                "feat_key": feat_key,
                "backbone": backbone,
                "manipulation": k,
                **v
            }
            for k, v in basic_metrics.items()
        ])

        metrics_file = os.path.join(
            evals_path,
            f"svd_metrics_{backbone}_{feat_key}.parquet"
        )

        df.to_parquet(metrics_file, index=False)

        csv_file = os.path.join(
            evals_path,
            f"svd_table_{backbone}_{feat_key}.csv"
        )

        df.to_csv(csv_file, index=False)

        print("Saved:", csv_file)

df_svs = pd.DataFrame(all_svs)

svs_csv_file = os.path.join(
    evals_path,
    "svd_singular_values_all_backbones_feats.parquet"
)

df_svs.to_parquet(svs_csv_file, index=False)

