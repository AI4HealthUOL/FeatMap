import re
import os
import sys
import time
import torch
import numpy as np
import yaml

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

sys.path.insert(0, "../..")
from train_mapping import MappingModel

"""Extraction of singular value decomposition (SVD) matrices from the weight matrics of the trained linear mapping models"""

def get_best_checkpoint(checkpoint_dir, checkpoint_prefix):
    try:
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith(checkpoint_prefix) and f.endswith(".ckpt")
        ]

        checkpoint_files.sort(
            key=lambda x: int(re.search(r"-v(\d+)", x).group(1))
            if re.search(r"-v(\d+)", x) else 0,
            reverse=True,
        )

        if not checkpoint_files:
            return None

        return os.path.join(checkpoint_dir, checkpoint_files[0])

    except FileNotFoundError:
        return None


def load_model(
    dataset,
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
):
    spatial_size = int(num_feature_vectors**0.5) if num_feature_vectors else 9

    model_config = {
        "model_type": model_type,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "spatial_size": spatial_size,
        **(model_params or {}),
    }

    model = MappingModel(model_config).to(device)

    checkpoint_dir = os.path.join(
        model_path,
        feature_subdir,
        dataset,
        model_type,
        target_manipulation
    )

    checkpoint_prefix = f"{model_type}_{feature_key}_normalized_{normalized}_mapping_model"

    model_checkpoint_path = get_best_checkpoint(
        checkpoint_dir, checkpoint_prefix)

    print(
        f"Loading mapping checkpoint: {model_checkpoint_path} (expected input_dim={input_dim}, spatial_size={spatial_size})"
    )
    state = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])

    model.eval()

    return model

def extract_weight_matrix(model):
    with torch.no_grad():
        return model.model.weight.detach().cpu().numpy()


def compute_svd(W):
    U, S, Vh = np.linalg.svd(
        W.astype(np.float32),
        full_matrices=False
    )
    return U, S, Vh


def resolve_feature_subdir(group_name):
    if group_name.startswith("qwen"):
        return "qwen_gs_1_infsteps_10"
    return "direct"


dataset = "STANFORD_CARS"
device = torch.device("cpu")
model_type = "linear"

model_path = os.path.expandvars("$WORK/models/")
output_dir = os.path.join(model_path, "svd_analysis")
os.makedirs(output_dir, exist_ok=True)

with open("../../../config/test_mapping.yaml", "r") as f:
    cfg = yaml.safe_load(f)

target = cfg["target_manipulations"][dataset]
backbones = cfg["extractor_name"]
feature_cfgs_all = cfg["feature_keys"]
model_params_all = cfg["models"]

for backbone in backbones:

    feature_cfgs = feature_cfgs_all[backbone]
    extractor_model_path = os.path.join(model_path, backbone)

    for feature_key, feat_cfg in feature_cfgs.items():

        input_dim = feat_cfg["input_dim"]
        output_dim = feat_cfg["output_dim"]
        spatial_size = feat_cfg["spatial_size"]
        num_feature_vectors = spatial_size * spatial_size

        svd_results = {}
        start_time = time.time()

        for group_name, group_cfg in target.items():

            allowed_models = [m for m in group_cfg["models"] if m.startswith("linear")]
            manipulations = group_cfg["manipulations"]
            feature_subdir = resolve_feature_subdir(group_name)

            for short_manip in manipulations:

                for model_type in allowed_models:

                    if backbone == "swinv2" or feature_key == "feat3":
                        normalized = False
                    else:
                        normalized = True

                    model_params = model_params_all.get(model_type, {})

                    print(f"Loading {backbone}/{group_name}/{short_manip}/{model_type}")

                    model = load_model(
                                dataset,
                                extractor_model_path,
                                normalized,
                                feature_subdir,
                                short_manip,
                                model_type,
                                input_dim,
                                output_dim,
                                model_params or {},
                                device,
                                feature_key,
                                num_feature_vectors=num_feature_vectors,
                            )

                    W = extract_weight_matrix(model)
                    U, S, Vh = compute_svd(W)

                    key_prefix = f"{dataset}_{backbone}_{group_name}_{short_manip}_linear_{feature_key}"

                    svd_results[key_prefix] = {
                        "U": U,
                        "S": S,
                        "Vh": Vh,
                    }

                    del model
                    torch.cuda.empty_cache()

        svd_file = os.path.join(
            output_dir,
            f"svd_{dataset}_{backbone}_{feature_key}.npz"
        )

        np.savez_compressed(svd_file, **svd_results)

        print(f"SVD saved: {svd_file}")
        print(f"Elapsed: {time.time() - start_time:.2f}s")