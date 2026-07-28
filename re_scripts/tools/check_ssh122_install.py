#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
settings_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "settings.json"
settings = json.loads(settings_path.read_text(encoding="utf-8"))

expected_agents = {
    "reasoner": {"max_model_len": 12288, "max_tokens": 896},
    "perceiver": {"max_model_len": 8192, "max_tokens": 512},
    "verifier": {"max_model_len": 4096, "max_tokens": 256},
}
for role, expected in expected_agents.items():
    actual = settings["agents"][role]
    for key, value in expected.items():
        assert actual.get(key) == value, f"{role}.{key}: {actual.get(key)!r} != {value!r}"

t = settings["trajectory"]
expected_trajectory = {
    "max_repair_attempts": 3,
    "question_similarity_threshold": 0.9,
    "enable_perceiver_focus_crop": True,
    "focus_crop_long_side": 1024,
    "focus_crop_padding_ratio": 0.02,
    "map_crop_coordinates": True,
    "classification_use_full_and_crop": True,
    "classification_crop_only_max_area": 0.0,
    "classification_reject_class_claims": False,
    "grounding_coordinate_crop_only": True,
    "reject_unsolicited_coordinates": True,
    "perceiver_fail_closed": True,
    "reasoner_fail_closed": True,
    "max_crop_coordinate_coverage": 0.94,
}
for key, value in expected_trajectory.items():
    assert t.get(key) == value, f"trajectory.{key}: {t.get(key)!r} != {value!r}"

launcher = (ROOT / "data_layer/official_socratic/02_run_official_generation.sh").read_text(encoding="utf-8")
required_exports = [
    "ALS_FOCUS_CROP_LONG_SIDE",
    "ALS_CLASSIFICATION_USE_FULL_AND_CROP",
    "ALS_GROUNDING_COORDINATE_CROP_ONLY",
    "ALS_REJECT_UNSOLICITED_COORDINATES",
    "ALS_PERCEIVER_FAIL_CLOSED",
    "ALS_REASONER_FAIL_CLOSED",
]
for name in required_exports:
    assert f"export {name}" in launcher, f"launcher does not export {name}"

adapter = (ROOT / "data_layer/official_socratic/local_api_adapter/utils.py").read_text(encoding="utf-8")
required_markers = [
    'CLASSIFICATION_USE_FULL_AND_CROP',
    'GROUNDING_COORDINATE_CROP_ONLY',
    'REJECT_UNSOLICITED_COORDINATES',
    'PERCEIVER_FAIL_CLOSED',
    'REASONER_FAIL_CLOSED',
    'for repair_index in range(1, MAX_REPAIR_ATTEMPTS + 1)',
    'crop_only_coordinate',
    'full_and_crop_classification',
]
for marker in required_markers:
    assert marker in adapter, f"adapter marker missing: {marker}"

print("[PASS] re_scripts_ssh1.2.2 configuration and source checks passed")
print(f"[INFO] settings: {settings_path}")
print("[INFO] Reasoner remains text-only; images are attached only in the Perceiver branch")
print("[INFO] Classification: full image + ROI; Grounding coordinate turn: ROI only")
