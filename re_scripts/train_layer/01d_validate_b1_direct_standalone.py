#!/usr/bin/env python3
"""Validate a Socratic-independent B1-Direct-Standalone dataset.

This gate accepts the four Direct QA modes introduced by v1.2.1 and refuses
Socratic rows, malformed JSON contracts, missing images, mixed modes, and an
optimizer-step count below the stable settings threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import count_optimizer_steps, load_settings, settings_from_cli, write_json  # noqa: E402

MODES = {
    "legacy_class_roi_obb",
    "all_image_json_obb",
    "all_image_json_hbb",
    "single_ref_json_obb",
}


def _assistant_payload(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 2:
        return ""
    content = str(messages[-1].get("content") or "")
    return content.rsplit("</think>", 1)[-1].strip()


def _numbers(values: Any, expected: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == expected
        and all(isinstance(value, (int, float)) for value in values)
    )


def _validate_payload(mode: str, payload: str) -> list[str]:
    issues: list[str] = []
    if mode == "legacy_class_roi_obb":
        if "FINAL_DETECTIONS" not in payload or "END_DETECTIONS" not in payload:
            issues.append("legacy_detection_markers_missing")
        return issues
    try:
        value = json.loads(payload)
    except Exception as exc:
        return [f"assistant_json_invalid:{type(exc).__name__}:{exc}"]
    if not isinstance(value, dict):
        return ["assistant_json_root_not_object"]
    if mode == "all_image_json_obb":
        objects = value.get("objects")
        if not isinstance(objects, list) or not objects:
            issues.append("objects_not_nonempty_list")
        else:
            for index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    issues.append(f"object_{index}_not_object")
                    continue
                if not isinstance(obj.get("category"), str) or not obj["category"].strip():
                    issues.append(f"object_{index}_category_invalid")
                if not _numbers(obj.get("obb_8"), 8):
                    issues.append(f"object_{index}_obb8_invalid")
    elif mode == "all_image_json_hbb":
        boxes = value.get("bbox_2d")
        categories = value.get("categories")
        if not isinstance(boxes, list) or not isinstance(categories, list):
            issues.append("bbox_or_categories_not_list")
        elif len(boxes) != len(categories) or not boxes:
            issues.append(f"bbox_category_length_mismatch:{len(boxes)}!={len(categories)}")
        else:
            for index, box in enumerate(boxes):
                if not _numbers(box, 4):
                    issues.append(f"bbox_{index}_invalid")
                if not isinstance(categories[index], str) or not categories[index].strip():
                    issues.append(f"category_{index}_invalid")
    elif mode == "single_ref_json_obb":
        if not isinstance(value.get("category"), str) or not value["category"].strip():
            issues.append("category_invalid")
        if not _numbers(value.get("obb_8"), 8):
            issues.append("obb8_invalid")
    else:
        issues.append(f"unsupported_mode:{mode}")
    return issues


def validate(path: Path, expected_mode: str) -> tuple[int, list[str], dict[str, int]]:
    if not path.is_file():
        return 0, [f"missing:{path}"], {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return 0, ["root_not_nonempty_list"], {}
    issues: list[str] = []
    task_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        if len(issues) >= 300:
            break
        if not isinstance(row, dict):
            issues.append(f"{index}:row_not_object")
            continue
        messages = row.get("messages") or []
        images = row.get("images") or []
        if len(messages) < 2 or not images:
            issues.append(f"{index}:missing_messages_or_images")
            continue
        if sum(str(message.get("content") or "").count("<image>") for message in messages) != len(images):
            issues.append(f"{index}:image_tag_count_mismatch")
        for image in images:
            if not Path(str(image)).is_file():
                issues.append(f"{index}:missing_image:{image}")
        metadata = row.get("metadata") or {}
        if metadata.get("official_socraticagent") or "trajectory_bucket" in metadata:
            issues.append(f"{index}:socratic_row_not_allowed")
        mode = str(metadata.get("direct_qa_mode") or "legacy_class_roi_obb")
        if mode not in MODES:
            issues.append(f"{index}:unsupported_direct_mode:{mode}")
        if mode != expected_mode:
            issues.append(f"{index}:mode_mismatch:{mode}!={expected_mode}")
        task = str(metadata.get("task") or "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
        payload_issues = _validate_payload(mode, _assistant_payload(row))
        issues.extend(f"{index}:{issue}" for issue in payload_issues)
        expected_objects = metadata.get("objects_in_sample")
        if expected_objects is not None and mode != "legacy_class_roi_obb":
            try:
                parsed = json.loads(_assistant_payload(row))
                if mode == "all_image_json_obb":
                    actual = len(parsed.get("objects") or [])
                elif mode == "all_image_json_hbb":
                    actual = len(parsed.get("bbox_2d") or [])
                else:
                    actual = 1
                if int(expected_objects) != actual:
                    issues.append(f"{index}:objects_in_sample_mismatch:{actual}!={expected_objects}")
            except Exception:
                pass
    return len(rows), issues, task_counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    expected_mode = str(settings.get("data_conversion", {}).get("direct_qa_mode", "legacy_class_roi_obb"))
    if expected_mode not in MODES:
        raise SystemExit(f"unsupported direct_qa_mode in settings: {expected_mode}")
    root = Path(settings["paths"]["dota128_llamafactory_root"])
    files = {
        "train": root / "dota128_direct_train_official.json",
        "val": root / "dota128_direct_val_official.json",
    }
    report: dict[str, Any] = {
        "schema_version": "b1_direct_standalone_gate_v1_2_1",
        "mode": expected_mode,
        "socratic_required": False,
        "files": {},
    }
    failed = False
    for split, path in files.items():
        count, issues, tasks = validate(path, expected_mode)
        report["files"][split] = {
            "path": str(path),
            "samples": count,
            "tasks": tasks,
            "issues": issues[:300],
        }
        failed = failed or bool(issues)
    training = settings["training"]
    steps = count_optimizer_steps(
        report["files"]["train"]["samples"],
        training["epochs"],
        training["per_device_train_batch_size"],
        training["gradient_accumulation_steps"],
        training["world_size"],
    )
    minimum = int(training.get("min_optimizer_steps", 1))
    report["estimated_optimizer_steps"] = steps
    report["minimum_optimizer_steps"] = minimum
    if steps < minimum:
        report.setdefault("critical", []).append(
            f"optimizer_steps_below_minimum:{steps}<{minimum}"
        )
        failed = True
    report["passed"] = not failed
    report_path = root / "b1_direct_standalone_validation.json"
    write_json(report_path, report)
    print(f"[B1 DIRECT STANDALONE VALIDATE] {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  mode={expected_mode}")
    print(f"  train={report['files']['train']['samples']} val={report['files']['val']['samples']}")
    print(f"  optimizer_steps={steps} minimum={minimum}")
    print(f"  report={report_path}")
    if not report["passed"]:
        for split, item in report["files"].items():
            for issue in item["issues"][:20]:
                print(f"  {split}: {issue}")
        for issue in report.get("critical", []):
            print(f"  critical: {issue}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
