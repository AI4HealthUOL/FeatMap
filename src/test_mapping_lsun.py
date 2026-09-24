import os
import json
import yaml
import time
from PIL import Image


import numpy as np
import torch
import timm


from torch.utils.data import DataLoader


from prepare_datasets.feature_dataset_impl import (
    PairedFeatureDataset,
    DatasetNormalize,
    test_collate_fn,
)


from FeatInv.featinv_reconstructor_conv import InputReconstructorConv
from FeatInv.featinv_reconstructor_swinv2 import InputReconstructorSwinV2
from FeatInv.featinv_reconstructor_dinov3 import InputReconstructorDinoV3


from dinov3_extractor import (
    DINOv3FeatureExtractor,
    DEFAULT_MODEL_NAME,
)


from eval_functions import Evaluator
from eval_helpers import process_features, process_image
from model_implementations import load_model, apply_spatial_transform
from utils import (
    get_img_path_by_id,
    set_seed,
    resolve_feature_subdir,
    get_shared_ids,
    filter_pairs_by_ids,
    sample_ids,
)


set_seed(42)


from timm.models.swin_transformer_v2 import SwinTransformerV2


class SwinV2StagedFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,  # full model
        )
        # Ensure we can access stages
        assert isinstance(self.backbone, SwinTransformerV2)


    def forward_features(self, x: torch.Tensor, stage: int):
        """
        Return features from the given SwinV2 stage (0–3) as [B, C, H, W].
        """
        x = self.backbone.patch_embed(x)


        # stages are in self.backbone.layers, indexed 0..3
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)


            if i == stage:
                # Normalize layout to [B, C, H, W]
                if x.ndim == 3:
                    # [B, HW, C] -> [B, C, H, W]
                    B, HW, C = x.shape
                    H = W = int(HW ** 0.5)
                    x = x.permute(0, 2, 1).view(B, C, H, W)
                elif x.ndim == 4:
                    # Could be [B, C, H, W] or [B, H, W, C]
                    B, C, H, W = x.shape
                    # Heuristic: if last dim looks like channels, permute
                    if C < x.shape[-1]:
                        # Assume [B, H, W, C]
                        x = x.permute(0, 3, 1, 2)
                    # Now treat as [B, C, H, W]
                else:
                    raise RuntimeError(
                        f"Unexpected feature tensor ndim={x.ndim}, shape={tuple(x.shape)}"
                    )


                return x.contiguous()


        # If we exit the loop (e.g. stage >= num_layers), treat final x similarly
        if x.ndim == 3:
            B, HW, C = x.shape
            H = W = int(HW ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        elif x.ndim == 4:
            B, C, H, W = x.shape
            if C < x.shape[-1]:
                x = x.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(
                f"Unexpected feature tensor ndim={x.ndim}, shape={tuple(x.shape)}"
            )


        return x.contiguous()


    def forward(self, x):
        return self.backbone(x)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-key",
        type=str,
        default=None,
        help="Feature key to evaluate on (e.g. feat0, feat1). If None, evaluate all.",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=None,
        help="Extractor to evaluate (e.g. convnext, swinv2, dinov3). If None, use config.",
    )
    return parser.parse_args()



def load_pairs_and_meta(extractor_name, dataset, dataset_path, feature_subdir, manipulation):
    base_root = os.path.join(dataset_path, dataset)


    feature_dir = os.path.join(
        base_root,
        f"{extractor_name}_features",
        "augmented_test",
        feature_subdir,
    )


    meta_path = os.path.join(feature_dir, "meta.jsonl")


    row_to_meta = {}
    with open(meta_path) as f:
        for line in f:
            m = json.loads(line)
            row_to_meta[m["row_idx"]] = m


    pairs = np.load(os.path.join(feature_dir, "pairs.npy"), allow_pickle=True)
    pairs = pairs[pairs[:, 2] == manipulation]


    return pairs, row_to_meta, feature_dir



def get_backbone(name, device):
    if name == "convnext":
        return timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            features_only=True,
        ).eval().to(device)


    if name == "swinv2":
        return SwinV2StagedFeatureExtractor(
            "swinv2_base_window12to24_192to384_22kft1k",
            pretrained=True,
        ).eval().to(device)


    if name == "dinov3":
        backbone = DINOv3FeatureExtractor(
            model_name=DEFAULT_MODEL_NAME,
            hf_token=os.environ.get("HF_TOKEN"),
            device=device,
        )


        return backbone


    raise ValueError(f"Unknown backbone name: {name}")



def run_eval_loop(test_loader, step_fn):
    with torch.inference_mode():
        for batch in test_loader:
            step_fn(batch)



def run_spatial_baseline(*args, **kwargs):
    (
        test_loader,
        target_manipulation,
        extractor_eval_path,
        feature_subdir,
        dataset,
        feature_key,
        row_to_meta,
        evaluator,
        reconstructor,
        test_dataset,
        normalized,
        normalizer,
        target_recon_paths,
        loss_name,
    ) = args


    print(">> Running spatial-only baseline")


    def step(batch):
        orig_feats, _, _, _, _, orig_indices, _ = batch
        orig_feats = orig_feats.cuda() if torch.cuda.is_available() else orig_feats


        spatial = apply_spatial_transform(
            orig_feats.clone(), target_manipulation)


        for i, idx in enumerate(orig_indices):
            meta = row_to_meta[idx]
            orig_id = meta["original_id"]


            feat = spatial[i]
            if normalized and normalizer is not None:
                feat = normalizer.denormalize(feat)


            img = process_features(feat.cpu().numpy(), reconstructor)


            mode = f"spatial_only_{target_manipulation}"


            save_dir = os.path.join(
                extractor_eval_path,
                f"{mode}/{feature_subdir}/{dataset}/{feature_key}/{target_manipulation}",
            )
            os.makedirs(save_dir, exist_ok=True)


            save_path = os.path.join(save_dir, f"{orig_id}_spatial.png")
            img.save(save_path)


            evaluator.generate_evaluation(
                mode=mode,
                dataset=dataset,
                original_id=orig_id,
                target_manipulation=target_manipulation,
                original_img_path=meta["path"],
                manipulated_img_path=get_img_path_by_id(
                    row_to_meta, orig_id, target_manipulation
                ),
                mapped_img_path=save_path,
                target_reconstructed_img_path=target_recon_paths.get(orig_id),
                eval_save_dir=save_dir,
                feature_key=feature_key,
                showFig=False,
                loss_name=loss_name,
            )


    run_eval_loop(test_loader, step)



def run_mapping_models(*args, **kwargs):
    (
        allowed_models,
        config,
        dataset,
        extractor,
        extractor_model_path,
        normalized,
        normalizer,
        feature_subdir,
        target_manipulation,
        device,
        feature_key,
        input_dim,
        output_dim,
        num_feature_vectors,
        test_loader,
        row_to_meta,
        evaluator,
        reconstructor,
        test_dataset,
        target_recon_paths,
        extractor_eval_path,
        apply_flag,
        loss_name,
    ) = args


    for model_type in allowed_models:
        model_params = config["models"].get(model_type, {})


        model = load_model(
            dataset=dataset,
            extractor_name=extractor,
            model_path=extractor_model_path,
            normalized=normalized,
            feature_subdir=feature_subdir,
            target_manipulation=target_manipulation,
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            model_params=model_params or {},
            device=device,
            feature_key=feature_key,
            loss_name=loss_name,
            num_feature_vectors=num_feature_vectors,
            apply_transform=apply_flag,
        )


        if model is None:
            print(
                "DEBUG: load_model returned None for",
                dataset, extractor, feature_key, target_manipulation, model_type,
            )
            continue


        def step(batch):
            orig_feats, _, _, _, _, orig_indices, _ = batch
            orig_feats = orig_feats.to(device)
            mapped = model(orig_feats)


            for i, idx in enumerate(orig_indices):
                meta = row_to_meta[idx]
                orig_id = meta["original_id"]


                feat = mapped[i]
                if normalized and normalizer is not None:
                    feat = normalizer.denormalize(feat)


                img = process_features(feat.cpu().numpy(), reconstructor)


                mode = (
                    f"{model_type}_{target_manipulation}_loss_{loss_name}"
                    if apply_flag
                    else f"{model_type}_loss_{loss_name}"
                )


                save_dir = os.path.join(
                    extractor_eval_path,
                    f"{mode}_{normalized}/{feature_subdir}/{dataset}/{feature_key}/{target_manipulation}",
                )
                os.makedirs(save_dir, exist_ok=True)


                save_path = os.path.join(save_dir, f"{orig_id}_mapped.png")
                img.save(save_path)


                evaluator.generate_evaluation(
                    mode=mode,
                    dataset=dataset,
                    original_id=orig_id,
                    target_manipulation=target_manipulation,
                    original_img_path=meta["path"],
                    manipulated_img_path=get_img_path_by_id(
                        row_to_meta, orig_id, target_manipulation
                    ),
                    mapped_img_path=save_path,
                    target_reconstructed_img_path=target_recon_paths.get(orig_id),
                    eval_save_dir=save_dir,
                    feature_key=feature_key,
                    showFig=False,
                    loss_name=loss_name,
                )


        run_eval_loop(test_loader, step)



def main(feature_key_filter, extractor_filter=None):
    start = time.time()


    with open("../config/test_mapping_lsun.yaml") as f:
        config = yaml.safe_load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_path = os.path.expandvars(config["model_path"])
    norm_path = os.path.expandvars(config["norm_params_path"])
    dataset_path = os.path.expandvars(config["dataset_path"])
    eval_path = os.path.expandvars(config["evals_path"])


    batch_size = config["batch_size"]
    subset_size = config.get("subset_size")


    allowed_extractors = config.get("extractor_name", ["convnext", "swinv2", "dinov3"])
    if extractor_filter is not None:
        allowed_extractors = [extractor_filter]


    for dataset in config["target_manipulations"]:
        groups = config["target_manipulations"][dataset]


        for group_name, group_cfg in groups.items():
            feature_subdir = resolve_feature_subdir(group_name)
            apply_flag = group_cfg.get("apply_transform", False)
            allowed_models = group_cfg["models"]


            for manipulation in group_cfg["manipulations"]:
                loss_functions = config.get("loss_functions", ["mse"])


                # Load pairs/meta for each extractor
                data_by_extractor = {}


                for extractor in allowed_extractors:
                    try:
                        pairs, meta, root = load_pairs_and_meta(
                            extractor,
                            dataset,
                            dataset_path,
                            feature_subdir,
                            manipulation,
                        )
                        data_by_extractor[extractor] = (pairs, meta, root)
                    except Exception as e:
                        print(
                            f"SKIP: cannot load pairs/meta for extractor={extractor}, "
                            f"dataset={dataset}, feature_subdir={feature_subdir}, "
                            f"manipulation={manipulation}: {e}"
                        )
                        # Skip this extractor for this combination
                        continue


                if not data_by_extractor:
                    print(
                        f"SKIP: no extractor has valid pairs/meta for "
                        f"dataset={dataset}, feature_subdir={feature_subdir}, "
                        f"manipulation={manipulation}"
                    )
                    continue


                # Determine shared ID set across extractors
                if len(allowed_extractors) == 1:
                    extractor = allowed_extractors[0]
                    if extractor not in data_by_extractor:
                        continue
                    _, meta, _ = data_by_extractor[extractor]
                    all_ids = [m["original_id"] for m in meta.values()]
                    selected = set(sample_ids(all_ids, subset_size or len(all_ids)))
       
                    for ex in allowed_extractors:
                        if ex not in data_by_extractor:
                            continue
                        p, m, r = data_by_extractor[ex]
                        data_by_extractor[ex] = (
                            filter_pairs_by_ids(p, m, selected),
                            m,
                            r,
                        )
                else:
                    # Build lists for get_shared_ids
                    conv_p = conv_m = swin_p = swin_m = dino_p = dino_m = None


                    if "convnext" in data_by_extractor:
                        conv_p, conv_m, _ = data_by_extractor["convnext"]
                    if "swinv2" in data_by_extractor:
                        swin_p, swin_m, _ = data_by_extractor["swinv2"]
                    if "dinov3" in data_by_extractor:
                        dino_p, dino_m, _ = data_by_extractor["dinov3"]

                    shared = get_shared_ids(conv_p, conv_m, swin_p, swin_m, dino_p, dino_m)
                    selected = set(sample_ids(shared, subset_size or len(shared)))
                           

                    for ex in allowed_extractors:
                        if ex not in data_by_extractor:
                            continue
                        p, m, r = data_by_extractor[ex]
                        data_by_extractor[ex] = (
                            filter_pairs_by_ids(p, m, selected),
                            m,
                            r,
                        )


                # Loop over extractors
                for extractor in allowed_extractors:
                    if extractor not in data_by_extractor:
                        continue


                    pairs, meta, root = data_by_extractor[extractor]


                    extractor_eval_path = os.path.join(eval_path, extractor)
                    os.makedirs(extractor_eval_path, exist_ok=True)


                    feature_cfgs = config["feature_keys"][extractor]


                    if feature_key_filter is not None:
                        if feature_key_filter not in feature_cfgs:
                            print(
                                f"SKIP: feature_key {feature_key_filter} not found "
                                f"in {list(feature_cfgs.keys())} for extractor={extractor}"
                            )
                            continue
                        feature_cfgs = {
                            feature_key_filter: feature_cfgs[feature_key_filter]
                        }


                    for feature_key, feat_cfg in feature_cfgs.items():
                        input_dim = feat_cfg["input_dim"]
                        output_dim = feat_cfg["output_dim"]
                        spatial_size = feat_cfg["spatial_size"]
                        normalized = feat_cfg["normalize_features"]


                        num_vec = spatial_size * spatial_size


                        # Create evaluator per feature_key to reduce peak memory
                        evaluator = Evaluator(eval_path, extractor)


                        # Reconstructor + backbone (wrapped)
                        try:
                            if extractor == "convnext":
                                backbone = get_backbone("convnext", device)
                                reconstructor = InputReconstructorConv(backbone, feature_key)
                            elif extractor == "swinv2":
                                backbone = get_backbone("swinv2", device)
                                reconstructor = InputReconstructorSwinV2(backbone, feature_key)
                            elif extractor == "dinov3":
                                backbone = get_backbone("dinov3", device)
                                reconstructor = InputReconstructorDinoV3(
                                    feature_model=backbone,
                                    feature_key=feature_key,
                                )
                            else:
                                raise ValueError(f"Unknown extractor: {extractor}")
                        except Exception as e:
                            print(
                                f"SKIP: cannot initialize backbone/reconstructor for "
                                f"extractor={extractor}, feature_key={feature_key}: {e}"
                            )
                            # Clean up evaluator as well
                            del evaluator
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue


                        test_dataset = PairedFeatureDataset(
                            path_prefix=root,
                            feature_key=feature_key,
                            pairs=pairs,
                            manipulation=manipulation,
                        )


                        normalizer = None


                        if normalized:
                            extractor_norm_path = os.path.join(
                                norm_path,
                                extractor,
                            )


                            expected_norm_file = os.path.join(
                                extractor_norm_path,
                                dataset,
                                f"resized_{feature_key}.pt",
                            )


                            print("Evaluation normalization root:", norm_path)
                            print("Evaluation extractor:", extractor)
                            print("Evaluation normalization directory:", extractor_norm_path)
                            print("Expected normalization file:", expected_norm_file)
                            print("Normalization file exists:", os.path.isfile(expected_norm_file))


                            if not os.path.isfile(expected_norm_file):
                                print(
                                    f"SKIP: normalization file missing for "
                                    f"extractor={extractor}, dataset={dataset}, "
                                    f"feature_key={feature_key}: {expected_norm_file}"
                                )
                                # Free heavy objects before continuing
                                del evaluator, backbone, reconstructor, test_dataset
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue


                            try:
                                normalizer = DatasetNormalize(
                                    dataset=test_dataset,
                                    source_dataset=dataset,
                                    manip="resized",
                                    feat_key=feature_key,
                                    norm_params_path=extractor_norm_path,
                                    recalc_norm_params=False,
                                )
                                print(
                                    "Loaded normalizer:",
                                    "mean=", tuple(normalizer.mean.shape),
                                    "std=", tuple(normalizer.std.shape),
                                )
                            except Exception as e:
                                print(
                                    f"SKIP: cannot load normalizer for "
                                    f"extractor={extractor}, dataset={dataset}, "
                                    f"feature_key={feature_key}: {e}"
                                )
                                del evaluator, backbone, reconstructor, test_dataset
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue


                        # Precompute target reconstructions
                        target_recon_paths = {}


                        for orig_id in selected:
                            try:
                                target_img_path = None
                                for m in meta.values():
                                    if (
                                        m["original_id"] == orig_id
                                        and m["manipulation"] == manipulation
                                    ):
                                        target_img_path = m["path"]
                                        break


                                if target_img_path is None:
                                    continue


                                if extractor == "dinov3":
                                    # Dino-specific: reconstruct from target features
                                    target_pair = None
                                    for pair in pairs:
                                        if meta[pair[0]]["original_id"] == orig_id:
                                            target_pair = pair
                                            break


                                    if target_pair is None:
                                        continue


                                    target_idx = target_pair[1]
                                    target_feat = test_dataset.backend.get_feat(target_idx)


                                    if normalized and normalizer is not None:
                                        target_feat = normalizer.denormalize(target_feat)


                                    target_img = reconstructor.reconstruct(target_feat)
                                else:
                                    img = Image.open(target_img_path).convert("RGB")
                                    target_img = process_image(img, reconstructor, extractor)


                                target_eval_dir = os.path.join(
                                    extractor_eval_path,
                                    f"{feature_subdir}/{dataset}/{feature_key}/{manipulation}/targets",
                                )
                                os.makedirs(target_eval_dir, exist_ok=True)


                                target_save_path = os.path.join(
                                    target_eval_dir,
                                    f"{orig_id}_target_recon.png",
                                )


                                # target_img may be a list/tuple for Dino
                                if isinstance(target_img, (list, tuple)):
                                    img_to_save = target_img[0]
                                    if isinstance(img_to_save, np.ndarray):
                                        img_to_save = Image.fromarray(img_to_save)
                                    img_to_save.save(target_save_path)
                                else:
                                    if isinstance(target_img, np.ndarray):
                                        target_img = Image.fromarray(target_img)
                                    target_img.save(target_save_path)


                                target_recon_paths[orig_id] = target_save_path


                            except Exception as e:
                                print(
                                    f"SKIP: error while precomputing target reconstruction "
                                    f"for extractor={extractor}, dataset={dataset}, "
                                    f"feature_key={feature_key}, orig_id={orig_id}: {e}"
                                )
                                # Continue with other IDs instead of failing everything
                                continue


                        if not target_recon_paths:
                            print(
                                f"SKIP: no valid target reconstructions for "
                                f"extractor={extractor}, dataset={dataset}, "
                                f"feature_key={feature_key}, manipulation={manipulation}"
                            )
                            del evaluator, backbone, reconstructor, test_dataset
                            if normalizer is not None:
                                del normalizer
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue


                        try:
                            test_loader = DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=lambda batch: test_collate_fn(batch, normalizer=normalizer),
                            )
                        except Exception as e:
                            print(
                                f"SKIP: cannot create DataLoader for "
                                f"extractor={extractor}, dataset={dataset}, "
                                f"feature_key={feature_key}: {e}"
                            )
                            del evaluator, backbone, reconstructor, test_dataset
                            if normalizer is not None:
                                del normalizer
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue


                        if apply_flag:
                            try:
                                run_spatial_baseline(
                                    test_loader,
                                    manipulation,
                                    extractor_eval_path,
                                    feature_subdir,
                                    dataset,
                                    feature_key,
                                    meta,
                                    evaluator,
                                    reconstructor,
                                    test_dataset,
                                    normalized,
                                    normalizer,
                                    target_recon_paths,
                                    "mse_median_cos",
                                )
                            except Exception as e:
                                print(
                                    f"SKIP: spatial baseline failed for "
                                    f"extractor={extractor}, dataset={dataset}, "
                                    f"feature_key={feature_key}, manipulation={manipulation}: {e}"
                                )


                        for loss_name in loss_functions:
                            print(f"\nEvaluating loss: {loss_name}\n")
                            try:
                                run_mapping_models(
                                    allowed_models,
                                    config,
                                    dataset,
                                    extractor,
                                    model_path,
                                    normalized,
                                    normalizer,
                                    feature_subdir,
                                    manipulation,
                                    device,
                                    feature_key,
                                    input_dim,
                                    output_dim,
                                    num_vec,
                                    test_loader,
                                    meta,
                                    evaluator,
                                    reconstructor,
                                    test_dataset,
                                    target_recon_paths,
                                    extractor_eval_path,
                                    apply_flag,
                                    loss_name,
                                )
                            except Exception as e:
                                print(
                                    f"SKIP: mapping models evaluation failed for "
                                    f"extractor={extractor}, dataset={dataset}, "
                                    f"feature_key={feature_key}, loss={loss_name}: {e}"
                                )


                        # Explicitly free everything for this feature_key
                        del backbone, reconstructor, evaluator, test_dataset
                        if normalizer is not None:
                            del normalizer
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()


                # Free memory between manipulations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


    print(f"Total time: {time.time() - start:.2f}s")



if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    main(
        feature_key_filter=args.feature_key,
        extractor_filter=args.extractor,
    )
    print(f"Total time: {time.time() - start_time:.2f}s")