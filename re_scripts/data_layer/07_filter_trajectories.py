#!/usr/bin/env python3
"""Audit trajectories with both semantic and deterministic geometry gates."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import (  # noqa: E402
    extract_final_text,
    hbb_from_points,
    hbb_iou,
    infer_and_convert_points,
    load_settings,
    normalize_class_name,
    parse_obb_output,
    read_jsonl,
    settings_from_cli,
    write_json,
)


def trace_text(row: dict[str, Any]) -> str:
    parts = []
    for turn in row.get("rounds", []):
        if turn.get("control_round"):
            continue
        parts += [
            f"Reasoning: {turn.get('thinking', '')}",
            f"Visual question: {turn.get('visual_question', '')}",
            f"Visual observation: {turn.get('observation', '')}",
        ]
    if row.get("final_thinking"):
        parts.append(f"Reasoning: {row['final_thinking']}")
    return "\n".join(x for x in parts if x.strip())


def _coord_mode(row: dict[str, Any]) -> str:
    target = row.get("coordinate_target")
    return {
        "pixel_obb": "pixel",
        "norm100_obb": "normalized_0_100",
        "norm1000_obb": "normalized_0_1000",
    }.get(target, "auto")


def _expanded_center_gate(pred_hbb: list[float], gt_hbb: list[float], ratio: float) -> bool:
    px = (pred_hbb[0] + pred_hbb[2]) / 2
    py = (pred_hbb[1] + pred_hbb[3]) / 2
    gx1, gy1, gx2, gy2 = gt_hbb
    margin_x = max(1.0, (gx2 - gx1) * ratio)
    margin_y = max(1.0, (gy2 - gy1) * ratio)
    return gx1 - margin_x <= px <= gx2 + margin_x and gy1 - margin_y <= py <= gy2 + margin_y


def geometry_audit(row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    gate = settings["trajectory"].get("geometry_gate", {})
    task = row.get("task")
    generated = extract_final_text(row.get("generated_final", ""))
    gt_class = normalize_class_name(row.get("class_name") or str(row.get("gt", "")).split("|", 1)[0])
    result: dict[str, Any] = {
        "enabled": bool(gate.get("enabled", True)),
        "task": task,
        "class_ok": False,
        "parse_ok": False,
        "hbb_iou": 0.0,
        "center_gate": False,
        "pass": False,
        "reason": "not_evaluated",
    }
    if not result["enabled"]:
        result.update({"pass": True, "reason": "disabled"})
        return result

    if task == "ref_classification":
        pred_class = normalize_class_name(generated)
        result.update(
            {
                "pred_class": pred_class,
                "gt_class": gt_class,
                "class_ok": pred_class == gt_class and pred_class is not None,
                "parse_ok": pred_class is not None,
            }
        )
        result["pass"] = result["class_ok"]
        result["reason"] = "ok" if result["pass"] else "canonical_class_mismatch"
        return result

    if task != "ref_grounding_obb":
        result.update({"pass": True, "reason": "non_ref_task"})
        return result

    width = int(row.get("image_width", 0))
    height = int(row.get("image_height", 0))
    if width <= 0 or height <= 0:
        result["reason"] = "missing_image_size"
        return result
    predictions = parse_obb_output(generated, width, height, _coord_mode(row))
    if not predictions:
        # Some models omit the FINAL_DETECTIONS wrapper. parse_obb_output already
        # accepts a single plain line, so reaching this branch means no usable OBB.
        result["reason"] = "no_parseable_obb"
        return result
    gt_points = row.get("obb_pixel") or row.get("obj_corner_pixel")
    if not isinstance(gt_points, list) or len(gt_points) != 4:
        result["reason"] = "missing_gt_obb"
        return result
    gt_hbb = hbb_from_points(gt_points)

    best = None
    for pred in predictions:
        pred_hbb = hbb_from_points(pred["points"])
        score = hbb_iou(pred_hbb, gt_hbb)
        candidate = (score, pred, pred_hbb)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    score, pred, pred_hbb = best
    class_ok = pred["class_name"] == gt_class
    center_ok = _expanded_center_gate(
        pred_hbb,
        gt_hbb,
        float(gate.get("expanded_gt_ratio", 0.35)),
    )
    min_iou = float(gate.get("min_hbb_iou_for_teacher_force", 0.1))
    require_center = bool(gate.get("require_center_in_expanded_gt", True))
    passed = class_ok and score >= min_iou and (center_ok or not require_center)
    result.update(
        {
            "parse_ok": True,
            "class_ok": class_ok,
            "pred_class": pred["class_name"],
            "gt_class": gt_class,
            "hbb_iou": round(float(score), 6),
            "center_gate": center_ok,
            "pred_points_pixel": pred["points"],
            "pred_coordinate_mode": pred.get("coordinate_mode_detected"),
            "pass": passed,
            "reason": "ok" if passed else (
                "canonical_class_mismatch" if not class_ok else
                "hbb_iou_below_gate" if score < min_iou else
                "predicted_center_outside_expanded_gt"
            ),
        }
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--split", default="train")
    ap.add_argument("--input", help="Optional trajectory JSONL; defaults to custom engine output")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    raw = Path(args.input) if args.input else Path(settings["paths"]["trajectory_raw_dir"]) / f"{args.split}_trajectories.jsonl"
    rows = read_jsonl(raw, skip_bad=True)
    out = Path(settings["paths"]["trajectory_audit_dir"])
    out.mkdir(parents=True, exist_ok=True)
    strict, relaxed, rejected = [], [], []
    reasons = Counter()
    trajectory = settings["trajectory"]

    for row in rows:
        why = []
        rounds = int(row.get("perception_rounds", 0))
        verifier = row.get("verifier") or {}
        repairs = row.get("format_repairs") or []
        geometry = geometry_audit(row, settings)
        if not row.get("success"):
            why.append(f"generation:{row.get('error')}")
        if not verifier.get("pass"):
            why.append(f"verifier:{verifier.get('reason', 'failed')}")
        if not geometry.get("pass"):
            why.append(f"geometry:{geometry.get('reason')}")
        if trajectory.get("reject_on_format_repair", False) and repairs:
            why.append("format_repair_used")
        bucket = None
        if (
            not why
            and rounds >= int(trajectory["min_perception_rounds_strict"])
            and float(verifier.get("score", 0)) >= float(trajectory["strict_min_verifier_score"])
        ):
            bucket = "strict"
        elif (
            not why
            and rounds == 1
            and trajectory.get("allow_single_round_if_verified", True)
            and float(verifier.get("score", 0)) >= float(trajectory["single_round_min_verifier_score"])
        ):
            bucket = "single_round_verified"
        else:
            if rounds < 1:
                why.append("no_visual_observation")
            elif rounds < int(trajectory["min_perception_rounds_strict"]):
                why.append(f"too_few_perception_rounds:{rounds}")

        audit = {
            "id": row.get("id"),
            "task": row.get("task"),
            "query": row.get("query"),
            "gt": row.get("gt"),
            "generated_final": row.get("generated_final"),
            "bucket": bucket,
            "reasons": why,
            "perception_rounds": rounds,
            "verifier": verifier,
            "geometry_audit": geometry,
            "format_repairs": repairs,
            "trace": trace_text(row),
            "image_path": row.get("image_path"),
            "image_width": row.get("image_width"),
            "image_height": row.get("image_height"),
            "class_name": row.get("class_name"),
            "obb_pixel": row.get("obb_pixel") or row.get("obj_corner_pixel"),
            "teacher_forced_final": row.get("teacher_forced_final"),
            "prompt_profile": row.get("prompt_profile"),
            "engine": row.get("engine"),
        }
        if bucket == "strict":
            strict.append(audit)
        elif bucket == "single_round_verified":
            relaxed.append(audit)
        else:
            rejected.append(audit)
            for reason in why:
                reasons[reason] += 1

    for name, data in (
        ("accepted_strict.json", strict),
        ("accepted_single_round.json", relaxed),
        ("rejected_trajectory_audit.json", rejected),
    ):
        write_json(out / name, data)
    report = {
        "input_file": str(raw),
        "input": len(rows),
        "strict": len(strict),
        "single_round_verified": len(relaxed),
        "rejected": len(rejected),
        "rejection_reasons": dict(reasons.most_common()),
        "policy": trajectory,
    }
    write_json(out / "trajectory_quality_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
