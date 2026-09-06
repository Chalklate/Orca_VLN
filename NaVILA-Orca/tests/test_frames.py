import base64
import io

import numpy as np
import pytest
from PIL import Image

from navila_orca.frames import (
    encode_images_jpeg_base64,
    sample_history,
    sample_history_with_step_ids,
    to_rgb_image,
)
from navila_orca.contracts import RenderFrame


def _frame(value: int) -> Image.Image:
    return Image.fromarray(np.full((6, 8, 3), value, dtype=np.uint8), mode="RGB")


def _value(image: Image.Image) -> int:
    return int(np.asarray(image)[0, 0, 0])


def test_short_history_is_left_padded_to_exactly_eight():
    sampled = sample_history([_frame(10), _frame(20), _frame(30)])
    assert len(sampled) == 8
    assert [_value(image) for image in sampled] == [0, 0, 0, 0, 0, 10, 20, 30]


def test_long_history_uses_exact_navila_uniform_indices_and_latest():
    sampled = sample_history([_frame(index) for index in range(12)])
    assert [_value(image) for image in sampled] == [0, 1, 3, 4, 6, 7, 9, 11]


def test_history_sidecar_reports_source_steps_and_padding():
    frames = [
        RenderFrame(
            step_id=index,
            sim_time_s=index * 0.02,
            camera_id="ego",
            rgb=np.full((6, 8, 3), index, dtype=np.uint8),
        )
        for index in (10, 20, 30)
    ]
    _, step_ids = sample_history_with_step_ids(frames)
    assert step_ids == [None, None, None, None, None, 10, 20, 30]


def test_float_rgb_conversion_matches_benchmark_scaling_rule():
    image = to_rgb_image(np.full((2, 3, 3), 0.5, dtype=np.float32))
    assert np.asarray(image).dtype == np.uint8
    assert np.all(np.asarray(image) == 127)


def test_jpeg_base64_encoding_produces_eight_decodable_rgb_images():
    encoded = encode_images_jpeg_base64([_frame(index * 10) for index in range(8)])
    assert len(encoded) == 8
    for payload in encoded:
        decoded = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        assert decoded.size == (8, 6)


def test_empty_history_and_wrong_request_size_are_rejected():
    with pytest.raises(ValueError):
        sample_history([])
    with pytest.raises(ValueError):
        encode_images_jpeg_base64([_frame(0)])
