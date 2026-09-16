"""NaVILA-compatible RGB conversion, history sampling, and JPEG encoding."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from .contracts import RenderFrame


NUM_VIDEO_FRAMES = 8


def brighten_image(image: Image.Image, factor: float) -> Image.Image:
    """Return an RGB image with brightness multiplied by ``factor``.

    A factor of ``1.0`` preserves the image. Values above ``1.0`` lift the
    image before it is encoded for the VLM; highlights may clip at the upper
    end of the RGB range.
    """

    if factor <= 0.0:
        raise ValueError("brightness factor must be positive")
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(float(factor))


def brighten_images(
    images: Sequence[Image.Image], factor: float
) -> list[Image.Image]:
    """Brighten a VLM image batch while preserving its frame count."""

    return [brighten_image(image, factor) for image in images]


def to_rgb_image(image: Image.Image | RenderFrame | np.ndarray) -> Image.Image:
    """Convert a supported frame to a detached PIL RGB image.

    Float arrays follow the original benchmark rule: values whose maximum is
    at most one are multiplied by 255; all other arrays are clipped directly
    to the uint8 range.
    """

    if isinstance(image, RenderFrame):
        return image.to_pil()
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"image array must have shape (H, W, 3|4), got {array.shape}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("image array must be non-empty and finite")
    if array.dtype != np.uint8:
        if float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return Image.fromarray(array, mode="RGB")


def sample_history(
    images: Sequence[Image.Image | RenderFrame | np.ndarray],
) -> list[Image.Image]:
    """Return NaVILA's exact eight-frame, full-history uniform sample.

    Histories shorter than eight frames are left-padded with black images.  For
    longer histories, seven uniformly spaced historical frames are selected
    with ``floor(i * (N - 1) / 7)`` and the latest frame is appended.
    """

    sampled, _ = sample_history_with_step_ids(images)
    return sampled


def sample_history_with_step_ids(
    images: Sequence[Image.Image | RenderFrame | np.ndarray],
) -> tuple[list[Image.Image], list[int | None]]:
    """Return the eight sampled images and their source physics step IDs.

    ``None`` identifies a synthetic black padding frame.  The step-ID sidecar
    is used by the collection recorder to prove which simulator observations
    were sent to NaVILA; it does not change the model input.
    """

    if not images:
        raise ValueError("cannot sample an empty image history")
    frames = [to_rgb_image(image) for image in images]
    step_ids: list[int | None] = [
        _frame_step_id(image) for image in images
    ]
    while len(frames) < NUM_VIDEO_FRAMES:
        frames.insert(0, Image.new("RGB", frames[-1].size, (0, 0, 0)))
        step_ids.insert(0, None)

    count = len(frames)
    indices = [int(index * (count - 1) / 7) for index in range(7)]
    sampled = [frames[index] for index in indices]
    sampled.append(frames[-1])
    sampled_step_ids = [step_ids[index] for index in indices]
    sampled_step_ids.append(step_ids[-1])
    return sampled, sampled_step_ids


def _frame_step_id(image: Any) -> int | None:
    """Read a physics step ID when the source item carries one."""

    value = getattr(image, "step_id", None)
    return int(value) if value is not None else None


def encode_jpeg_base64(image: Image.Image | RenderFrame | np.ndarray) -> str:
    """Encode one RGB frame exactly as the benchmark TCP client does."""

    buffer = io.BytesIO()
    to_rgb_image(image).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_images_jpeg_base64(
    images: Sequence[Image.Image | RenderFrame | np.ndarray],
) -> list[str]:
    if len(images) != NUM_VIDEO_FRAMES:
        raise ValueError(
            f"NaVILA requests require exactly {NUM_VIDEO_FRAMES} images, got {len(images)}"
        )
    return [encode_jpeg_base64(image) for image in images]
