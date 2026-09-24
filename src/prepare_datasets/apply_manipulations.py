import time
import os
import re
import gc
import json
from PIL import Image
from torchvision import transforms
import torch
import multiprocessing
import yaml

from diffusers import DiffusionPipeline, QwenImageEditPlusPipeline

from huggingface_hub import login

from manipulations import (
    apply_rotation,
    apply_translation,
    apply_mirror,
    apply_noise,
    apply_mask,
    apply_perspective_transform,
    apply_hue_shift,
    apply_grayscale,
)


"""
Dataset augmentation pipeline for feature-space analysis.

This script generates manipulated versions of input images using:
1. Direct image transformations (geometric, photometric, masking)
2. Generative image editing via Qwen (semantic editing)

Pipeline:
- Load dataset images (train/test split)
- Apply selected manipulations
- Save augmented images using a consistent naming scheme:
    <ID>_<manipulation>.jpg

Outputs:
- augmented_train/
- augmented_test/

See config/apply_manipulations.yaml for configuration.
"""


def extract_leading_number(folder_name):
    match = re.match(r"(\d+)", folder_name)
    return int(match.group(1)) if match else float("inf")


def extract_numeric_id(filename):
    """
    Extracts numeric ID from filename.

    All image samples from the dataset have unique IDs.
    """
    match = re.search(r"(\d+)(?=\.\w+$)", filename)
    return match.group(1) if match else os.path.splitext(filename)[0]


def process_image_direct_manip(image_info, transform):
    """
    Applies a set of predefined direct manipulations to an image.

    Manipulations include:
    - geometric: rotation, translation, perspective
    - photometric: noise, hue shift, grayscale
    - structural: masking, mirroring

    Each manipulated image is saved using the naming convention:
        <base_name>_<manipulation>.jpg

    Args:
        image_info: Tuple containing:
            (image_path, output_dir, base_name, manipulation_cfg)
        transform: Preprocessing transform (resize + crop)

    Returns:
        str: Base image ID.
    """
    image_path, class_out_dir, base_name, manipulation_cfg = image_info

    direct_cfg = manipulation_cfg.get("direct", {})

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        # Resize to target size
        img = transform(img)

        # This is the original base image resized
        img.save(
            os.path.join(class_out_dir, base_name + "_resized.jpg"),
            format="JPEG",
        )

        for deg in direct_cfg.get("rotation", []):
            manipulated = apply_rotation(img, deg)
            manipulated.save(
                os.path.join(class_out_dir, f"{base_name}_rotation_{deg}.jpg"),
                format="JPEG",
            )

        for tx in direct_cfg.get("translation_x", []):
            for ty in direct_cfg.get("translation_y", []):
                manipulated = apply_translation(img, tx, ty)
                manipulated.save(
                    os.path.join(
                        class_out_dir, f"{base_name}_translation_{tx}_{ty}.jpg"
                    ),
                    format="JPEG",
                )

        for direction in direct_cfg.get("mirror", []):
            manipulated = apply_mirror(img, direction)
            manipulated.save(
                os.path.join(
                    class_out_dir, f"{base_name}_mirror_{direction}.jpg"),
                format="JPEG",
            )

        for amount in direct_cfg.get("added_noise", []):
            manipulated = apply_noise(img, amount)
            manipulated.save(
                os.path.join(class_out_dir, f"{base_name}_noise_{amount}.jpg"),
                format="JPEG",
            )

        for mask_name, mask_info in direct_cfg.get("masks", {}).items():
            polygon_points = mask_info["polygon"]
            color = mask_info["color"]
            manipulated = apply_mask(img, polygon_points, color)
            manipulated.save(
                os.path.join(
                    class_out_dir, f"{base_name}_mask_{mask_name}.jpg"),
                format="JPEG",
            )

        for idx, warp_points in enumerate(
            direct_cfg.get("perspective_transformations", {}).get("warp", [])
        ):
            manipulated = apply_perspective_transform(img, warp_points)
            manipulated.save(
                os.path.join(
                    class_out_dir, f"{base_name}_perspective_{idx}.jpg"),
                format="JPEG",
            )

        for hue in direct_cfg.get("color_adjustments", {}).get("hue_shift", []):
            manipulated = apply_hue_shift(img, hue)
            manipulated.save(
                os.path.join(
                    class_out_dir, f"{base_name}_hue_shift_{hue}.jpg"),
                format="JPEG",
            )

        if direct_cfg.get("color_adjustments", {}).get("grayscale", False):
            manipulated = apply_grayscale(img)
            manipulated.save(
                os.path.join(class_out_dir, f"{base_name}_grayscale.jpg"),
                format="JPEG",
            )

    return base_name


def process_image_qwen_manip(image_info, transform, qwen_prompts):
    """
    Applies generative image editing using Qwen.

    For each prompt:
    - generates edited versions of the input image
    - saves outputs with prompt-encoded filenames

    This enables semantic manipulations beyond simple pixel-level transforms.

    Args:
        image_info: Tuple containing:
            (image_path, output_dir, base_name, pipeline, parameters)
        transform: Preprocessing transform
        qwen_prompts (List[str]): List of editing prompts

    """
    (
        image_path,
        class_out_dir,
        base_name,
        qwen_pipeline,
        qwen_parameters,
    ) = image_info

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = transform(img)
        img.save(os.path.join(class_out_dir, base_name + "_resized.jpg"))
        (
            guidance_scale,
            num_inference_steps,
            true_cfg_scale,
            negative_prompt,
            generator,
            repeats_per_prompt,
        ) = qwen_parameters

        for prompt in qwen_prompts:
            inputs = {
                "image": [img],
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt or " ",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": repeats_per_prompt,
            }

            with torch.inference_mode():
                output = qwen_pipeline(**inputs)

            for idx, out_img in enumerate(output.images):
                save_name = f"{base_name}_qwen_{prompt.replace(' ', '_')}_{idx}.jpg"

                out_img.save(
                    os.path.join(
                        class_out_dir,
                        save_name,
                    )
                )
            # Running this for many images build up VRAM
            # Clearing made this possible here
            del output
            gc.collect()
            torch.cuda.empty_cache()


def get_tasks(
    input_dir,
    output_dir,
    sorted_classes,
    num_classes,
    run_direct,
    manipulation_cfg,
    imgs_per_class_qwen=None,
    qwen_pipeline=None,
    qwen_parameters=None,
):
    """Build the task list for each method and sort savepaths"""
    image_tasks_direct = []
    image_tasks_qwen = []
    selected_classes = (
        sorted_classes if num_classes == -1 else sorted_classes[:num_classes]
    )
    for class_folder in selected_classes:
        class_path = os.path.join(input_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        class_out_dir = os.path.join(output_dir, class_folder)
        os.makedirs(class_out_dir, exist_ok=True)
        if run_direct:
            class_out_dir_direct = os.path.join(class_out_dir, "direct")
            os.makedirs(class_out_dir_direct, exist_ok=True)

        if qwen_pipeline is not None:
            class_out_dir_qwen = os.path.join(
                class_out_dir,
                f"qwen_gs_"
                f"{int(manipulation_cfg.get('qwen', {}).get('guidance_scale', 1.0))}_"
                f"infsteps_{manipulation_cfg.get('qwen', {}).get('num_inference_steps', 40)}",
            )
            os.makedirs(class_out_dir_qwen, exist_ok=True)

        for idx, image_name in enumerate(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_name)
            base_name = extract_numeric_id(image_name)
            if run_direct:
                image_tasks_direct.append(
                    (image_path, class_out_dir_direct, base_name, manipulation_cfg)
                )

            if qwen_pipeline is not None and (
                imgs_per_class_qwen is None or idx < imgs_per_class_qwen
            ):
                image_tasks_qwen.append(
                    (
                        image_path,
                        class_out_dir,
                        base_name,
                        qwen_pipeline,
                        qwen_parameters,
                    )
                )
    return image_tasks_direct, image_tasks_qwen


def main():
    # All config settings
    # Which manipulations, which datasets, train, test split
    with open("../../config/apply_manipulations.yaml") as f:
        manipulation_cfg = yaml.safe_load(f)

    # Currently supported: STANFORD_CARS
    datasets = manipulation_cfg.get("datasets", [])
    if not datasets:
        datasets = [manipulation_cfg["dataset"]]
    target_img_size = manipulation_cfg["target_img_size"]

    # Enable/Disable either of the manipulation methods
    run_direct = manipulation_cfg["direct"]["enabled"]
    run_qwen = manipulation_cfg["qwen"]["enabled"]

    dataset_path = os.path.expandvars(manipulation_cfg.get("dataset_path", ""))

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        num_classes = manipulation_cfg["num_classes"]
        transform = transforms.Compose(
            [
                transforms.Resize(target_img_size),
                transforms.CenterCrop(target_img_size),
            ]
        )
        if dataset == "STANFORD_CARS":
            root_dir = os.path.join(dataset_path, "STANFORD_CARS/")

        # Any dataset used here needs to use this folder structure
        # root_dir/images/train, root_dir/images/test
        # In each train/test directory should be the classfolders
        # Example from the Cars dataset: datasets/STANFORD_CARS/images/train/Audi_100_Wagon_1994/N.jpg
        # Where N is a **unique** id per image
        train_dir = os.path.join(root_dir, "images/train")
        test_dir = os.path.join(root_dir, "images/test")

        # All manipulated images are then stored in seperate folders
        # using the same classfolder structure
        train_output_dir = os.path.join(root_dir, "images/augmented_train")
        os.makedirs(train_output_dir, exist_ok=True)
        test_output_dir = os.path.join(root_dir, "images/augmented_test")
        os.makedirs(test_output_dir, exist_ok=True)

        sorted_train_classes = sorted(os.listdir(
            train_dir), key=extract_leading_number)
        sorted_test_classes = sorted(os.listdir(
            test_dir), key=extract_leading_number)

        # Optionally limits the number of images used for the imade editing models
        imgs_per_class = (
            manipulation_cfg["qwen"]["imgs_per_class"] if run_qwen else None
        )

        qwen_pipeline = None
        qwen_parameters = None
        qwen_prompts = None
        imgs_per_class_qwen = None
        height = width = target_img_size

        # Qwen requires a valid Huggingface token
        if run_qwen:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            token = os.environ.get("HF_TOKEN")

            if token is None or token.startswith("hf_***"):
                raise ValueError(
                    "HF_TOKEN environment variable must be set with a valid Hugging Face token. "
                    "Get one from https://huggingface.co/settings/tokens and export it: "
                    "export HF_TOKEN='hf_your_token_here'"
                )
            login(token=token)

            # Parameters based on the Huggingface examples from:
            # https://huggingface.co/Qwen/Qwen-Image-Edit-2511
            qwen_model_id = manipulation_cfg["qwen"].get(
                "model_id", "Qwen/Qwen-Image-Edit-2511"
            )
            qwen_num_inference_steps = manipulation_cfg["qwen"].get(
                "num_inference_steps", 40
            )
            qwen_guidance_scale = manipulation_cfg["qwen"].get(
                "guidance_scale", 1.0)
            qwen_true_cfg_scale = manipulation_cfg["qwen"].get(
                "true_cfg_scale", 4.0)
            qwen_negative_prompt = manipulation_cfg["qwen"].get(
                "negative_prompt", " ")
            qwen_repeats = manipulation_cfg["qwen"].get(
                "repeats_per_prompt", 1)
            qwen_prompts = manipulation_cfg["qwen"]["prompts"][dataset]
            qwen_seed = manipulation_cfg["qwen"].get("seed", 0)
            imgs_per_class_qwen = manipulation_cfg["qwen"].get(
                "imgs_per_class", None)

            base_repo = "Qwen/Qwen-Image-Edit-2511"
            n_gpus = torch.cuda.device_count()

            # Qwen-Image-Edit-2511 is a large model (~57 GB)
            # For this use case 2 Nvidia L40 GPUs were required
            # Running this on a large enough single GPU improved inference speed when available
            if n_gpus > 1:
                max_memory = {
                    i: int(torch.cuda.get_device_properties(
                        i).total_memory * 0.95)
                    for i in range(n_gpus)
                }

                max_memory["cpu"] = int(2 * 1024**3)

                qwen_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    qwen_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="balanced",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                )
            else:
                qwen_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    qwen_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                )

            qwen_pipeline.vae.enable_slicing()
            qwen_pipeline.vae.enable_tiling()

            torch.backends.cudnn.benchmark = True

            qwen_pipeline.set_progress_bar_config(disable=None)

            qwen_generator = torch.Generator()
            qwen_generator.manual_seed(qwen_seed)
            qwen_parameters = (
                qwen_guidance_scale,
                qwen_num_inference_steps,
                qwen_true_cfg_scale,
                qwen_negative_prompt,
                qwen_generator,
                qwen_repeats,
            )

        # This builds all combinations of methods+manipulations as task lists
        train_image_tasks_direct, train_image_tasks_qwen = (
            get_tasks(
                train_dir,
                train_output_dir,
                sorted_train_classes,
                num_classes,
                run_direct,
                manipulation_cfg,
                imgs_per_class,
                imgs_per_class_qwen,
                qwen_pipeline,
                qwen_parameters,
            )
        )
        test_image_tasks_direct, test_image_tasks_qwen = (
            get_tasks(
                test_dir,
                test_output_dir,
                sorted_test_classes,
                num_classes,
                run_direct,
                manipulation_cfg,
                5,
                5,
                qwen_pipeline,
                qwen_parameters,
            )
        )

        # Executes the tasks per method
        if run_direct:
            print("Applying direct transforms")
            start_time = time.time()
            # Direct manipulations dont require GPU use
            # multiprocessing greatly improves generation
            with multiprocessing.Pool(processes=8) as pool:
                pool.starmap(
                    process_image_direct_manip,
                    [(task, transform) for task in train_image_tasks_direct],
                )
                pool.starmap(
                    process_image_direct_manip,
                    [(task, transform) for task in test_image_tasks_direct],
                )
            end_time = time.time()
            print(
                f"Direct manipulations for train/test {dataset} with {num_classes} class(es) runtime: {end_time - start_time:.2f} seconds"
            )

        if run_qwen:
            start_time = time.time()

            for task in train_image_tasks_qwen:
                gc.collect()
                torch.cuda.empty_cache()
                process_image_qwen_manip(
                    task,
                    transform,
                    qwen_prompts,
                )

            train_qwen_time = time.time() - start_time

            start_time = time.time()

            for task in test_image_tasks_qwen:
                gc.collect()
                torch.cuda.empty_cache()
                process_image_qwen_manip(
                    task,
                    transform,
                    qwen_prompts,
                )

            test_qwen_time = time.time() - start_time

            print(
                f"\nQwen manipulations for train {dataset} "
                f"for nr tasks {len(train_image_tasks_qwen)} "
                f"runtime: {train_qwen_time:.2f} seconds"
            )

            print(
                f"Qwen manipulations for test {dataset} "
                f"for nr tasks {len(test_image_tasks_qwen)} "
                f"runtime: {test_qwen_time:.2f} seconds"
            )


if __name__ == "__main__":
    main()
