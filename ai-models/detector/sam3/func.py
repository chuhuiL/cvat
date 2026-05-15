# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""CVAT auto-annotation function that uses Segment Anything 3 (SAM3) from Meta.

SAM3 supports open-vocabulary segmentation from a text prompt. This function
runs SAM3 with a fixed text prompt (e.g. "Pupil") on every frame and emits one
CVAT label per text phrase.

Usage with the CVAT CLI:

    cvat-cli task auto-annotate <task_id> \
        --function-file ai-models/detector/sam3/func.py \
        --function-parameter model_id=str:facebook/sam3 \
        --function-parameter text=str:Pupil \
        --function-parameter device=str:cuda
"""

from __future__ import annotations

import cv2
import cvat_sdk.auto_annotation as cvataa
import numpy as np
import PIL.Image
import torch
from cvat_sdk.masks import encode_mask
from transformers import Sam3Model, Sam3Processor


class _Sam3DetectionFunction:
    """Open-vocabulary detector wrapping facebook/sam3."""

    def __init__(
        self,
        *,
        model_id: str = "facebook/sam3",
        text: str = "object",
        device: str = "cpu",
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        output_shape: str = "mask",
        poly_epsilon: float = 1.0,
    ) -> None:
        if output_shape not in ("mask", "polygon"):
            raise ValueError(f"output_shape must be 'mask' or 'polygon', got {output_shape!r}")

        self._device = torch.device(device)
        self._text = text
        self._score_threshold = float(score_threshold)
        self._mask_threshold = float(mask_threshold)
        self._output_shape = output_shape
        self._poly_epsilon = float(poly_epsilon)

        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._model = (
            Sam3Model.from_pretrained(model_id, torch_dtype=dtype).to(self._device).eval()
        )
        self._processor = Sam3Processor.from_pretrained(model_id)

        # One CVAT label per phrase; for now we ship a single phrase, but split on
        # "|" to allow simple multi-phrase prompts ("Pupil|Iris") if desired.
        self._phrases = [p.strip() for p in self._text.split("|") if p.strip()]
        if not self._phrases:
            raise ValueError("text prompt must contain at least one non-empty phrase")

        self.spec = cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(name, label_id, type=self._output_shape)
                for label_id, name in enumerate(self._phrases)
            ]
        )

    @torch.inference_mode()
    def detect(
        self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
    ) -> list[cvataa.DetectionAnnotation]:
        score_threshold = (
            context.conf_threshold if context.conf_threshold is not None else self._score_threshold
        )
        emit_polygon = (
            context.conv_mask_to_poly if self._output_shape == "mask" else self._output_shape == "polygon"
        )

        rgb = image.convert("RGB")
        annotations: list[cvataa.DetectionAnnotation] = []
        for label_id, phrase in enumerate(self._phrases):
            inputs = self._processor(images=rgb, text=phrase, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                mask_threshold=self._mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results["masks"]
            for det_idx in range(len(masks)):
                mask_np = masks[det_idx].detach().cpu().numpy().astype(np.uint8)
                if mask_np.shape != (rgb.height, rgb.width):
                    mask_np = cv2.resize(
                        mask_np, (rgb.width, rgb.height), interpolation=cv2.INTER_NEAREST
                    )
                if not mask_np.any():
                    continue

                if emit_polygon:
                    contours, _ = cv2.findContours(
                        mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not contours:
                        continue
                    largest = max(contours, key=cv2.contourArea)
                    approx = cv2.approxPolyDP(largest, epsilon=self._poly_epsilon, closed=True)
                    if approx.shape[0] < 3:
                        continue
                    annotations.append(
                        cvataa.polygon(label_id, approx.flatten().tolist())
                    )
                else:
                    annotations.append(cvataa.mask(label_id, encode_mask(mask_np.astype(bool))))

        return annotations


create = _Sam3DetectionFunction
