#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_layer.official_socratic.local_api_adapter import utils as adapter  # noqa: E402


def _classification_context() -> str:
    return (
        "<image>\nThe target is centered inside the coarse focus region "
        "[100,200,300,400] in normalized [0,1000] coordinates. "
        "Which canonical DOTA category is it?"
    )


def _grounding_context() -> str:
    return (
        "<image>\nThe target is inside the coarse focus region [100,200,300,400] "
        "in normalized [0,1000] coordinates. Locate the plane and return four "
        "clockwise OBB corners."
    )


def test_settings_generation_budgets_and_quality_thresholds() -> None:
    s = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
    assert s["agents"]["reasoner"]["max_tokens"] == 896
    assert s["agents"]["perceiver"]["max_tokens"] == 512
    assert s["agents"]["verifier"]["max_tokens"] == 256
    t = s["trajectory"]
    assert t["classification_use_full_and_crop"] is True
    assert t["classification_crop_only_max_area"] == 0.0
    assert t["grounding_coordinate_crop_only"] is True
    assert t["reject_unsolicited_coordinates"] is True
    assert t["perceiver_fail_closed"] is True
    assert t["reasoner_fail_closed"] is True
    assert t["debug_min_grounding_strict_rate"] == 0.30
    assert t["debug_min_classification_strict_rate"] == 0.70


def test_classification_keeps_full_scene_and_roi() -> None:
    full = (Image.new("RGB", (1000, 1000)), "png")
    crop = (Image.new("RGB", (512, 512)), "png")
    images, mode = adapter._select_perceiver_images(
        [full, crop],
        {"norm1000_bounds": [100, 200, 300, 400]},
        _classification_context(),
        "What visible shape and parts does the target have?",
    )
    assert mode == "full_and_crop_classification"
    assert images is not None and len(images) == 2


def test_grounding_coordinate_turn_uses_only_roi() -> None:
    full = (Image.new("RGB", (1000, 1000)), "png")
    crop = (Image.new("RGB", (512, 512)), "png")
    images, mode = adapter._select_perceiver_images(
        [full, crop],
        {"norm1000_bounds": [100, 200, 300, 400]},
        _grounding_context(),
        "What are the four clockwise OBB corners of the exact target?",
    )
    assert mode == "crop_only_coordinate"
    assert images is not None and len(images) == 1
    assert images[0] is crop


def test_crop_local_obb_maps_back_to_original_frame() -> None:
    crop_meta = {
        "norm1000_bounds": [100.0, 200.0, 300.0, 400.0],
        "pixel_bounds": [100, 200, 300, 400],
        "original_size": [1000, 1000],
        "crop_size": [512, 512],
    }
    result, normalized, reason = adapter._canonicalize_perceiver_coordinate_response(
        "obb_8=[200,200,800,200,800,800,200,800]",
        "obb",
        crop_meta,
    )
    assert reason == "ok"
    assert normalized
    assert result == "obb_8=[140,240,260,240,260,360,140,360]"


def test_entire_crop_and_unsolicited_coordinates_are_rejected() -> None:
    crop_meta = {
        "norm1000_bounds": [100.0, 200.0, 300.0, 400.0],
        "pixel_bounds": [100, 200, 300, 400],
        "original_size": [1000, 1000],
        "crop_size": [512, 512],
    }
    _, _, reason = adapter._canonicalize_perceiver_coordinate_response(
        "obb_8=[0,0,1000,0,1000,1000,0,1000]",
        "obb",
        crop_meta,
    )
    assert reason == "coordinate_copies_entire_crop"

    valid, reason = adapter._perceiver_response_status(
        "The target is visible. obb_8=[100,100,200,100,200,200,100,200]",
        "What visible parts does the exact target have?",
        _grounding_context(),
    )
    assert not valid
    assert reason == "unsolicited_coordinate_payload"


def test_class_name_in_useful_classification_evidence_is_not_hard_rejected() -> None:
    valid, reason = adapter._perceiver_response_status(
        "The plane-shaped target has a central fuselage, two lateral wings, and a tail.",
        "What structural parts are directly visible?",
        _classification_context(),
    )
    assert valid and reason == "ok"
