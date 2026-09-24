import os

import torch
from transformers import AutoModel


DEFAULT_MODEL_NAME = os.environ.get(
    "DINOV3_MODEL_NAME",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
)

DINOV3_MEAN = [0.485, 0.456, 0.406]
DINOV3_STD = [0.229, 0.224, 0.225]


# The values are one-based transformer-block indices used
# to index outputs.hidden_states.
DINO_STAGE_TO_BLOCK = {
    0: 3,
    1: 6,
    2: 9,
}

# Public stage ID for the final transformer representation.
DINO_FINAL_STAGE = 3


class DINOv3FeatureExtractor:
    """
    DINOv3 extractor supporting intermediate and final stages.

    Public stages:

        stage 0 -> transformer block 3
        stage 1 -> transformer block 6
        stage 2 -> transformer block 9
        stage 3 -> final transformer block

    All extracted feature maps have shape:

        (batch_size, hidden_size, height, width)

    The CLS token can optionally be returned as:

        (batch_size, hidden_size, 1, 1)
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        hf_token=None,
        device=None,
    ):
        hf_token = hf_token or os.environ.get("HF_TOKEN")

        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            AutoModel.from_pretrained(
                model_name,
                token=hf_token,
            )
            .eval()
            .to(self.device)
        )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.num_register_tokens = getattr(
            self.model.config,
            "num_register_tokens",
            4,
        )

        self.hidden_size = self.model.config.hidden_size

    @property
    def num_blocks(self):
        """
        Number of transformer blocks in the loaded model.
        """
        return self.model.config.num_hidden_layers

    @property
    def supported_stages(self):
        return tuple(
            sorted(
                set(DINO_STAGE_TO_BLOCK)
                | {DINO_FINAL_STAGE}
            )
        )

    def _resolve_stage(self, stage):
        """
        Convert a public stage ID to an internal hidden-state index.

        Returns:
            None for the final layer, otherwise the integer
            index into outputs.hidden_states.
        """
        if stage == DINO_FINAL_STAGE:
            return None

        if stage not in DINO_STAGE_TO_BLOCK:
            raise ValueError(
                f"Unknown DINO stage {stage}. "
                f"Supported stages are "
                f"{self.supported_stages}."
            )

        block = DINO_STAGE_TO_BLOCK[stage]

        if not 1 <= block <= self.num_blocks:
            raise ValueError(
                f"Stage {stage} maps to transformer block "
                f"{block}, but the loaded model has "
                f"{self.num_blocks} blocks."
            )

        return block

    def _tokens_to_features(self, tokens, stage):
        """
        Split CLS/register tokens from patch tokens and convert
        patch tokens to NCHW.
        """
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected tokens with shape (B, N, C), "
                f"got {tuple(tokens.shape)} for stage {stage}."
            )

        batch_size, sequence_length, hidden_size = (
            tokens.shape
        )

        expected_prefix_tokens = (
            1 + self.num_register_tokens
        )

        if sequence_length <= expected_prefix_tokens:
            raise RuntimeError(
                f"No patch tokens remain for stage {stage}. "
                f"Received sequence length {sequence_length} "
                f"with {expected_prefix_tokens} prefix tokens."
            )

        cls_token = tokens[:, 0]

        patch_tokens = tokens[
            :,
            expected_prefix_tokens:,
        ]

        num_patches = patch_tokens.shape[1]
        side = int(round(num_patches ** 0.5))

        if side * side != num_patches:
            raise RuntimeError(
                f"Patch-token count {num_patches} is not a "
                f"perfect square at stage {stage}. "
                f"Input resolution and patch size may be "
                f"incompatible."
            )

        patch_map = patch_tokens.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            hidden_size,
            side,
            side,
        ).contiguous()

        return cls_token, patch_map

    @torch.inference_mode()
    def forward_features(
        self,
        pixel_values,
        stage=None,
    ):
        """
        Extract CLS and patch-map features.

        Args:
            pixel_values:
                Normalized tensor of shape (B, 3, H, W).

            stage:
                Public stage ID.

                - None: final layer.
                - 0, 1, 2: configured intermediate layers.
                - 3: final layer.

        Returns:
            cls_token:
                Tensor of shape (B, hidden_size).

            patch_map:
                Tensor of shape (B, hidden_size, h, w).
        """
        pixel_values = pixel_values.to(
            self.device,
            non_blocking=True,
        )

        block = (
            None
            if stage is None
            else self._resolve_stage(stage)
        )

        if block is None:
            # Final output. This is the last transformer layer.
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=False,
            )
            tokens = outputs.last_hidden_state

        else:
            # Intermediate output. hidden_states[0] is the
            # embedding output; hidden_states[block] is the
            # output after transformer block `block`.
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )

            tokens = outputs.hidden_states[block]

        return self._tokens_to_features(
            tokens,
            stage="final" if stage is None else stage,
        )

    @torch.inference_mode()
    def extract(
        self,
        pixel_values,
        target_layers,
    ):
        """
        Extract patch feature maps for one or more public stages.

        Returns:
            Dict mapping stage ID to NCHW feature map.
        """
        target_layers = list(target_layers)

        if not target_layers:
            raise ValueError(
                "target_layers must not be empty."
            )

        invalid = set(target_layers) - set(
            self.supported_stages
        )

        if invalid:
            raise ValueError(
                f"Invalid DINO stages {sorted(invalid)}. "
                f"Supported stages are "
                f"{self.supported_stages}."
            )

        features = {}

        # Run the model only once when multiple stages are
        # requested. The final output is available alongside
        # all intermediate hidden states.
        needs_intermediate = any(
            stage != DINO_FINAL_STAGE
            for stage in target_layers
        )

        pixel_values = pixel_values.to(
            self.device,
            non_blocking=True,
        )

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=needs_intermediate,
        )

        for stage in target_layers:
            if stage == DINO_FINAL_STAGE:
                tokens = outputs.last_hidden_state
            else:
                block = DINO_STAGE_TO_BLOCK[stage]
                tokens = outputs.hidden_states[block]

            _, patch_map = self._tokens_to_features(
                tokens,
                stage=stage,
            )

            features[stage] = patch_map

        return features

    @torch.inference_mode()
    def get_hint(
        self,
        pixel_values,
        mode,
        stage=None,
    ):
        """
        Return either the CLS token or patch feature map.

        Args:
            mode:
                'cls' or 'featmap'.

            stage:
                None or DINO_FINAL_STAGE for the final layer,
                or an intermediate public stage ID.
        """
        cls_token, patch_map = self.forward_features(
            pixel_values,
            stage=stage,
        )

        if mode == "cls":
            return cls_token.unsqueeze(
                -1
            ).unsqueeze(
                -1
            )

        if mode == "featmap":
            return patch_map

        raise ValueError(
            f"Unknown mode {mode!r}; expected "
            "'cls' or 'featmap'."
        )

    def close(self):
        # No hooks are registered, so there is nothing to
        # explicitly remove.
        pass