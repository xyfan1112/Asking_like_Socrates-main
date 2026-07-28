#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import DOTA_CLASSES, load_settings, obb_iou, read_jsonl, settings_from_cli, write_json  # noqa: E402


def ap101(rec, prec):
    return float(np.mean([prec[rec >= r].max() if np.any(rec >= r) else 0 for r in np.linspace(0, 1, 101)])) if len(rec) else 0.0


def class_ap(gt, pred, cid, threshold):
    ground = [x for x in gt if x["class_id"] == cid]
    predictions = sorted([x for x in pred if x["class_id"] == cid], key=lambda x: x.get("confidence", 1), reverse=True)
    by_image = defaultdict(list)
    for item in ground:
        by_image[item["eval_key"]].append(item)
    matched = {key: set() for key in by_image}
    tp, fp = [], []
    for item in predictions:
        best = (0.0, None)
        for index, target in enumerate(by_image.get(item["eval_key"], [])):
            if index in matched[item["eval_key"]]:
                continue
            score = obb_iou(item["points"], target["points"])
            if score > best[0]:
                best = (score, index)
        if best[1] is not None and best[0] >= threshold:
            tp.append(1); fp.append(0); matched[item["eval_key"]].add(best[1])
        else:
            tp.append(0); fp.append(1)
    if not predictions:
        return 0.0
    tc, fc = np.cumsum(tp), np.cumsum(fp)
    return ap101(tc / max(len(ground), 1), tc / np.maximum(tc + fc, 1e-9))


def greedy_iou_matches(
    ground_truth: list[dict],
    predictions: list[dict],
) -> tuple[list[float], int]:
    """Return one IoU per GT (unmatched=0) and the number of extra predictions."""
    pairs = sorted(
        (
            (obb_iou(pred["points"], gt["points"]), pred_index, gt_index)
            for pred_index, pred in enumerate(predictions)
            for gt_index, gt in enumerate(ground_truth)
            if pred.get("class_id") == gt.get("class_id")
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    ious = [0.0] * len(ground_truth)
    for score, pred_index, gt_index in pairs:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        ious[gt_index] = float(score)
    return ious, max(0, len(predictions) - len(used_pred))


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
    expected = int(manifest.get("expected_rows", len(rows)))
    errors = sum(bool(r.get("error")) for r in rows)
    truncated = sum(str(r.get("finish_reason") or "").lower() == "length" for r in rows)
    format_failures = sum(not r.get("format_ok", False) for r in rows)
    incomplete = len(rows) != expected

    gt, pred = [], []
    for row in rows:
        eval_key = f"{row.get('query_id')}#run{row.get('run_id', 0)}"
        for item in row.get("ground_truth", []):
            gt.append({**item, "eval_key": eval_key})
        for item in row.get("predictions", []):
            pred.append({**item, "eval_key": eval_key})
    classes = sorted({x["class_id"] for x in gt})
    thresholds = np.arange(0.5, 0.96, 0.05)
    per_class = []
    for cid in classes:
        values = [class_ap(gt, pred, cid, float(t)) for t in thresholds]
        per_class.append({
            "class_id": cid, "class_name": DOTA_CLASSES[cid], "ap50": values[0], "map50_95": float(np.mean(values))
        })
    all_gt_ious: list[float] = []
    extra_predictions = 0
    negative_rows = 0
    correct_negative_rows = 0
    for row in rows:
        row_gt = row.get("ground_truth", [])
        row_pred = row.get("predictions", [])
        if not row_gt:
            negative_rows += 1
            correct_negative_rows += int(not row_pred)
        matched_ious, extras = greedy_iou_matches(row_gt, row_pred)
        all_gt_ious.extend(matched_ious)
        extra_predictions += extras

    status = (
        "INVALID"
        if incomplete
        else "DEGRADED"
        if errors > 0 or truncated > 0 or format_failures > 0
        else "VALID"
    )
    summary = {
        "status": status,
        "rows": len(rows), "expected_rows": expected, "incomplete": incomplete,
        "error_rows": errors, "truncated_rows": truncated, "format_failure_rows": format_failures,
        "gt_total": len(gt), "pred_total": len(pred),
        "map50": float(np.mean([x["ap50"] for x in per_class])) if per_class else 0.0,
        "map50_95": float(np.mean([x["map50_95"] for x in per_class])) if per_class else 0.0,
        "mean_iou_per_gt": float(np.mean(all_gt_ious)) if all_gt_ious else 0.0,
        "extra_predictions_after_matching": extra_predictions,
        "negative_query_rows": negative_rows,
        "negative_query_accuracy": correct_negative_rows / max(negative_rows, 1),
        "output_protocol_success_rate": sum(bool(r.get("format_ok")) and not r.get("error") for r in rows) / max(len(rows), 1),
        "note": (
            "Malformed rows are assigned zero predictions, so IoU/AP remain conservative. "
            "INVALID means rows are missing; DEGRADED means all rows exist but some API, "
            "truncation, or format failures occurred."
        ),
    }
    for threshold in settings["evaluation"].get("iou_thresholds", [0.5, 0.7]):
        key = str(threshold).replace(".", "_")
        true_positive = sum(value >= float(threshold) for value in all_gt_ious)
        false_negative = len(all_gt_ious) - true_positive
        # Every unmatched/low-IoU prediction is a false positive at this threshold.
        false_positive = 0
        for row in rows:
            row_gt = row.get("ground_truth", [])
            row_pred = row.get("predictions", [])
            pairs = sorted(
                (
                    (obb_iou(pred_item["points"], gt_item["points"]), pidx, gidx)
                    for pidx, pred_item in enumerate(row_pred)
                    for gidx, gt_item in enumerate(row_gt)
                    if pred_item.get("class_id") == gt_item.get("class_id")
                ),
                reverse=True,
            )
            used_p: set[int] = set()
            used_g: set[int] = set()
            for score, pidx, gidx in pairs:
                if score < float(threshold):
                    break
                if pidx not in used_p and gidx not in used_g:
                    used_p.add(pidx)
                    used_g.add(gidx)
            false_positive += len(row_pred) - len(used_p)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        summary[f"precision_iou_{key}"] = precision
        summary[f"recall_iou_{key}"] = recall
        summary[f"f1_iou_{key}"] = (
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    output = Path(args.output) if args.output else pred_path.with_name(pred_path.stem + "_metrics.json")
    write_json(output, {"summary": summary, "per_class": per_class, "manifest": manifest})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if incomplete and settings["evaluation"].get("fail_on_incomplete", True):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
