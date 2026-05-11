import json
import os
import numpy as np 

"""Util functions for repeated use"""

def get_shared_ids(pairs_a, meta_a, pairs_b, meta_b):
    a = {meta_a[int(p[0])]["original_id"] for p in pairs_a}
    b = {meta_b[int(p[0])]["original_id"] for p in pairs_b}
    return list(a & b)


def filter_pairs_by_ids(pairs, meta, ids):
    ids = set(ids)
    return np.array([
        p for p in pairs
        if meta[int(p[0])]["original_id"] in ids
    ])


def sample_ids(ids, k, seed=42):
    rng = np.random.default_rng(seed)
    k = min(k, len(ids))
    return rng.choice(ids, k, replace=False)

def get_mapping_run_paths(
    base_path,
    extractor_name,
    feature_subdir,
    dataset,
    model_type,
    manipulation,
    apply_transform
):
    group_type = "applied" if apply_transform else "learned"

    base = os.path.join(
        base_path,
        extractor_name,
        feature_subdir,
        dataset,
        group_type,
        model_type,
        manipulation,
    )

    return base
    
def set_seed(seed: int = 42, deterministic: bool = False):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def get_img_path_by_id(metadata, original_id, manipulation):
    """
    Retrieves the image path for a given original ID and manipulation.

    Args:
        metadata (Dict[int, dict]): Mapping from dataset index → metadata entry.
        original_id (str): ID of the original image.
        manipulation (str): Target manipulation type.

    Returns:
        str or None: Path to the corresponding image if found.
    """
    for _, info in metadata.items():
        if (
            info.get("original_id") == original_id
            and info.get("manipulation") == manipulation
        ):
            return info.get("path")
    return None

def resolve_feature_subdir(group_name):
    """
    Resolves the subdirectory name for feature storage based on
    the manipulation group.

    Args:
        group_name (str): Name of manipulation group (e.g., "qwen_*", "direct").

    Returns:
        str: Subdirectory name used in the dataset structure.
    """
    if group_name.startswith("qwen"):
        return "qwen_gs_1_infsteps_10"

    return "direct"

def extract_original_id(path: str):
    """
    Extracts the original image ID from a feature file path.

    All filenames follow the convention:
        <ID>_<manipulation>.jpg

    Args:
        path (str): File path to feature or image.

    Returns:
        str: Original image ID.
    """
    filename = os.path.basename(path)
    return filename.split("_")[0]


def build_id_to_indices(paths):
    """
    Builds a mapping from original image IDs to dataset indices.

    This is used to group multiple manipulated versions of the same
    original image.

    Args:
        paths (List[str]): List of file paths.

    Returns:
        Dict[str, List[int]]: Mapping from original ID → list of indices.
    """
    id_to_indices = {}
    for i, p in enumerate(paths):
        oid = extract_original_id(p)
        id_to_indices.setdefault(oid, []).append(i)
    return id_to_indices


def build_manip_indices(meta_path, manip):
    """
    Retrieves dataset indices corresponding to a specific manipulation.

    The metadata file (meta.jsonl) contains entries of the form:
        {"row_idx": int, "manipulation": str, ...}

    Args:
        meta_path (str): Path to metadata file.
        manip (str): Manipulation name to filter for.

    Returns:
        List[int]: Indices of samples with the given manipulation.
    """
    rows = []
    with open(meta_path, "r") as f:
        for line in f:
            meta = json.loads(line)
            if meta["manipulation"] == manip:
                rows.append(meta["row_idx"])
    return rows