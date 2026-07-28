#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main_layer"))

try:
    import openai  # noqa: F401
except ImportError:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object
    sys.modules["openai"] = fake_openai

from data_layer.official_socratic.local_api_adapter import utils as adapter  # noqa: E402
from data_layer.ref_semantics import resolve_reference_matches  # noqa: E402
from data_layer.trajectory_gates import (  # noqa: E402
    audit_trace_semantics,
    audit_trajectory_evidence,
)


def test_perceiver_context_and_question_guard() -> None:
    reasoner_query = """# User Query:
<image>
The target is centered inside the coarse focus region [100,200,300,400] in normalized [0,1000].
Which canonical DOTA category is it?

# Image Metadata:
- Modality: RGB / Visible (size: 682x682)

<thinking>old history must not be copied</thinking>
<question>What is visible?</question>
"""
    context = adapter._extract_reasoner_target_context(reasoner_query)
    assert "[100,200,300,400]" in context
    assert "682x682" in context
    assert "old history" not in context
    augmented = adapter._augment_perceiver_query(
        "What is the target's visible shape?",
        None,
        context,
    )
    assert "# Original target context" in augmented
    assert "# Current atomic visual question" in augmented
    assert "[100,200,300,400]" in augmented
    adapter._remember_reasoner_context(reasoner_query)
    assert adapter._current_reasoner_context()
    adapter._remember_reasoner_context("unexpected prompt without a user block")
    assert adapter._current_reasoner_context() == ""

    repeated, reason = adapter._reasoner_format_status(
        "<thinking>retry</thinking><question>Are any small vehicles in the lower-right area?</question>",
        False,
        ["Are there small vehicles in the lower-right area?"],
        False,
    )
    assert not repeated and reason.startswith("duplicate_or_similar_question")
    leading, reason = adapter._reasoner_format_status(
        "<thinking>ask</thinking><question>Is the target a storage tank?</question>",
        False,
        [],
        True,
    )
    assert not leading and reason == "classification_question_reveals_or_asks_class"
    neutral, reason = adapter._reasoner_format_status(
        "<thinking>ask</thinking><question>What boundary shape and internal markings are visible on the target?</question>",
        False,
        [],
        True,
    )
    assert neutral and reason == "ok"


def test_semantic_reference_resolver() -> None:
    descriptors = [
        {
            "object_index": 0,
            "class_name": "large vehicle",
            "center_pixel": [100, 100],
            "semantic_region": "upper-left area",
            "geometry_phrase_used": "strongly elongated and roughly horizontal",
            "semantic_extremes": ["leftmost"],
        },
        {
            "object_index": 1,
            "class_name": "large vehicle",
            "center_pixel": [150, 100],
            "semantic_region": "upper-left area",
            "geometry_phrase_used": "strongly elongated and roughly horizontal",
            "semantic_extremes": ["rightmost"],
        },
        {
            "object_index": 2,
            "class_name": "small vehicle",
            "center_pixel": [200, 100],
            "semantic_region": "upper-left area",
            "geometry_phrase_used": "elongated and roughly horizontal",
            "semantic_extremes": [],
        },
        {
            "object_index": 3,
            "class_name": "small vehicle",
            "center_pixel": [220, 100],
            "semantic_region": "upper-left area",
            "geometry_phrase_used": "elongated and roughly horizontal",
            "semantic_extremes": [],
        },
    ]
    ambiguous_anchor = {
        "class_name": "large vehicle",
        "region": "upper-left area",
        "geometry_phrase": "strongly elongated and roughly horizontal",
        "extreme": None,
        "anchor": {
            "class_name": "small vehicle",
            "relation": "left of",
            "require_unique_class": True,
            "max_distance_ratio": 0.35,
        },
    }
    assert resolve_reference_matches(ambiguous_anchor, descriptors, 1000, 1000) == []
    unique_extreme = {
        **ambiguous_anchor,
        "extreme": "leftmost",
        "anchor": None,
    }
    assert resolve_reference_matches(unique_extreme, descriptors, 1000, 1000) == [0]


def test_semantic_and_geometry_gates() -> None:
    classification_item = {
        "task": "ref_classification",
        "class_name": "large vehicle",
        "gt": "large vehicle",
    }
    classification_raw = {
        "loop_result": {
            "final_answer": "large vehicle",
            "chat_history": [
                {
                    "round": 1,
                    "R_response": "<thinking>x</thinking><question>Is the target a large vehicle?</question>",
                    "P_response": "The target appears to be a parking lot beside a building.",
                },
                {
                    "round": 2,
                    "R_response": "<thinking>x</thinking><question>Is this target a large vehicle?</question>",
                    "P_response": "The target is a truck.",
                },
            ],
        }
    }
    semantic = audit_trace_semantics(
        classification_item,
        classification_raw,
        {"trajectory": {"question_similarity_threshold": 0.82}},
    )
    assert not semantic["pass"]
    assert semantic["duplicate_question_trajectory"]
    assert semantic["classification_contradiction"]
    assert semantic["classification_leading_questions"]

    wrong_visual_claim = {
        "loop_result": {
            "final_answer": "roundabout",
            "chat_history": [
                {
                    "round": 1,
                    "R_response": (
                        "<thinking>x</thinking><question>"
                        "What boundary and connected-road pattern is visible around the target?"
                        "</question>"
                    ),
                    "P_response": (
                        "It seems to fit the description of a storage tank."
                    ),
                }
            ],
        }
    }
    wrong_claim_semantic = audit_trace_semantics(
        {
            "task": "ref_classification",
            "class_name": "roundabout",
            "gt": "roundabout",
        },
        wrong_visual_claim,
        {"trajectory": {"question_similarity_threshold": 0.82}},
    )
    assert wrong_claim_semantic["classification_contradiction"]
    assert "storage tank" in wrong_claim_semantic["classification_wrong_class_claims"]

    grounding_item = {
        "task": "ref_grounding_obb",
        "class_name": "ship",
        "gt": "ship|100,100,200,100,200,200,100,200",
        "image_width": 1000,
        "image_height": 1000,
        "coordinate_target": "pixel_obb",
        "obb_pixel": [[100, 100], [200, 100], [200, 200], [100, 200]],
    }
    grounding_raw = {
        "loop_result": {
            "final_answer": "ship|140,140,150,140,150,150,140,150",
            "chat_history": [
                {
                    "round": 1,
                    "P_response": "bbox_2d=[140,140,150,150]",
                }
            ],
        }
    }
    settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
    strict = audit_trajectory_evidence(grounding_item, grounding_raw, settings)
    assert strict["center_gate"] is True
    assert strict["iou_gate"] is False
    assert strict["pass"] is False
    relaxed_settings = deepcopy(settings)
    relaxed_settings["trajectory"]["geometry_gate"]["pass_if_iou_or_center"] = True
    relaxed = audit_trajectory_evidence(
        grounding_item,
        grounding_raw,
        relaxed_settings,
    )
    assert relaxed["pass"] is True


def test_debug_warn_is_nonzero() -> None:
    with tempfile.TemporaryDirectory(prefix="v433_debug_gate_") as td:
        temp = Path(td)
        settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
        settings["paths"]["pipeline_work_root"] = str(temp / "pipeline")
        agent_dir = temp / "pipeline/agent_inputs"
        agent_dir.mkdir(parents=True)
        agent = {
            "id": "x",
            "task": "ref_classification",
            "class_name": "ship",
            "gt": "ship",
        }
        (agent_dir / "train_agent_inputs.jsonl").write_text(
            json.dumps(agent) + "\n",
            encoding="utf-8",
        )
        settings_path = temp / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        raw = {
            "id": "x",
            "task": "ref_classification",
            "loop_result": {
                "success": False,
                "final_answer": "bridge",
                "chat_history": [],
                "error": None,
            },
        }
        raw_path = temp / "debug.jsonl"
        raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        api_path = temp / "api.jsonl"
        api_path.write_text(
            json.dumps(
                {
                    "event": "api_call",
                    "role": "perceiver",
                    "finish_reason": "stop",
                    "original_context_attached": True,
                    "focus_roi_attached": True,
                    "original_context_task": "ref_classification",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = temp / "audit.json"
        result = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "data_layer/official_socratic/02b_audit_debug_generation.py"
                ),
                "--settings",
                str(settings_path),
                "--raw",
                str(raw_path),
                "--api-log",
                str(api_path),
                "--output",
                str(report_path),
            ],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "WARN"
        assert report["api"]["perceiver_context_missing"] == 0
        assert report["api"]["perceiver_focus_roi_missing"] == 0


def main() -> None:
    test_perceiver_context_and_question_guard()
    test_semantic_reference_resolver()
    test_semantic_and_geometry_gates()
    test_debug_warn_is_nonzero()
    print("V4.3.3 QUALITY GATE TESTS PASS")


if __name__ == "__main__":
    main()
