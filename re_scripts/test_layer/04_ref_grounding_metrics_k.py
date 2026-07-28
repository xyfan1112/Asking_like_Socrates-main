#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import hbb_iou, load_settings, normalize_class_name, obb_iou, read_jsonl, settings_from_cli, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    pred_path = Path(args.pred)
    rows = read_jsonl(pred_path, skip_bad=True)
    manifest_path = pred_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected = int(manifest.get("expected_runs", len(rows)))
    expected_k = int(manifest.get("k", 1))
    by = defaultdict(list); per = []
    errors = sum(bool(r.get("error")) for r in rows)
    truncated = sum(str(r.get("finish_reason") or "").lower() == "length" for r in rows)
    for row in rows:
        if row.get("target_type") == "obb" and row.get("pred_obb") and row.get("gt_obb"):
            iou = obb_iou(row["pred_obb"], row["gt_obb"])
        else:
            iou = hbb_iou(row["pred_hbb"], row["gt_hbb"]) if row.get("pred_hbb") else 0.0
        gt_class = normalize_class_name(row.get("gt_class"))
        # VRSBench-Ref may not require a class label in the answer.
        class_required = gt_class is not None and row.get("dataset") == "dota_ref"
        class_ok = (normalize_class_name(row.get("pred_class")) == gt_class) if class_required else True
        item = {**row, "iou": float(iou), "class_correct": bool(class_ok)}
        by[row["id"]].append(item); per.append(item)
    distribution = Counter(len(x) for x in by.values()) if by else Counter()
    incomplete = len(rows) != expected or any(len(x) != expected_k for x in by.values())
    parse_failures = sum(not r.get("parse_ok", r.get("pred_hbb") is not None) for r in rows)
    status = (
        "INVALID"
        if incomplete
        else "DEGRADED"
        if errors > 0 or truncated > 0 or parse_failures > 0
        else "VALID"
    )
    summary = {
        "status": status,
        "runs": len(rows), "expected_runs": expected, "questions": len(by), "expected_k": expected_k,
        "runs_per_question_distribution": dict(sorted(distribution.items())), "incomplete": incomplete,
        "error_rows": errors, "truncated_rows": truncated,
        "parse_failure_rows": parse_failures,
        "parse_success_rate": sum(r.get("pred_hbb") is not None for r in rows) / max(len(rows), 1),
        "class_accuracy": sum(r["class_correct"] for r in per) / max(len(per), 1),
        "mean_iou_all_runs": sum(r["iou"] for r in per) / max(len(per), 1),
    }
    for threshold in settings["evaluation"]["iou_thresholds"]:
        key = str(threshold).replace(".", "_")
        localization = [int(r["iou"] >= threshold) for r in per]
        joint = [int(r["class_correct"] and r["iou"] >= threshold) for r in per]
        summary[f"localization_acc_iou_{key}"] = sum(localization) / max(len(localization), 1)
        summary[f"joint_class_iou_acc_{key}"] = sum(joint) / max(len(joint), 1)
        # Compatibility alias: unlike v4.3.1 this is geometry-only and is named explicitly above.
        summary[f"iou_at_{key}"] = summary[f"localization_acc_iou_{key}"]
        for label, field in (("localization", "iou"), ("joint", "joint")):
            qvalues = [
                [
                    int(r["iou"] >= threshold)
                    if field == "iou"
                    else int(r["class_correct"] and r["iou"] >= threshold)
                    for r in group
                ]
                for group in by.values()
            ]
            summary[f"avg_at_k_{label}_{key}"] = (
                sum(sum(v) / len(v) for v in qvalues) / max(len(qvalues), 1)
            )
            summary[f"conv_at_k_{label}_{key}"] = (
                sum(sum(v) >= math.ceil(len(v) / 2) for v in qvalues)
                / max(len(qvalues), 1)
            )
            summary[f"pass_at_k_{label}_{key}"] = (
                sum(any(v) for v in qvalues) / max(len(qvalues), 1)
            )

    groups: dict[str, dict[str, float | int]] = {}
    for group_name in sorted({str(r.get("ref_group", "all")) for r in per}):
        members = [r for r in per if str(r.get("ref_group", "all")) == group_name]
        groups[group_name] = {
            "runs": len(members),
            "mean_iou": sum(r["iou"] for r in members) / max(len(members), 1),
            "class_accuracy": sum(r["class_correct"] for r in members) / max(len(members), 1),
        }
        for threshold in settings["evaluation"]["iou_thresholds"]:
            key = str(threshold).replace(".", "_")
            groups[group_name][f"localization_acc_iou_{key}"] = (
                sum(r["iou"] >= threshold for r in members) / max(len(members), 1)
            )
    summary["by_ref_group"] = groups
    output = Path(args.output) if args.output else pred_path.with_name(pred_path.stem + "_metrics.json")
    write_json(output, {"summary": summary, "manifest": manifest, "per_run": per})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Grounding主指标：mIoU与IoU@0.5/0.7；Avg/Conv/Pass@K仅作为多次采样稳定性扩展。")
    if incomplete and settings["evaluation"].get("fail_on_incomplete", True):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
