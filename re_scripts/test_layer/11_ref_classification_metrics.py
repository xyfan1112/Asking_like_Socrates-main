#!/usr/bin/env python3
"""Classification accuracy, macro-F1, and confusion for DOTA128-Ref."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import (  # noqa: E402
    DOTA_CLASSES,
    load_settings,
    normalize_class_name,
    read_jsonl,
    settings_from_cli,
    write_json,
)


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
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    expected = int(manifest.get("expected_runs", len(rows)))
    incomplete = len(rows) != expected
    errors = sum(bool(row.get("error")) for row in rows)
    truncated = sum(
        str(row.get("finish_reason") or "").lower() == "length" for row in rows
    )
    parse_failures = sum(not row.get("parse_ok", False) for row in rows)
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    for row in rows:
        gt = normalize_class_name(row.get("gt_class")) or "invalid_gt"
        pred = normalize_class_name(row.get("pred_class")) or "__parse_failure__"
        confusion[(gt, pred)] += 1
        correct += int(gt == pred)

    per_class = []
    for class_name in DOTA_CLASSES:
        support = sum(count for (gt, _), count in confusion.items() if gt == class_name)
        if support == 0:
            continue
        tp = confusion[(class_name, class_name)]
        fp = sum(
            count
            for (gt, pred), count in confusion.items()
            if pred == class_name and gt != class_name
        )
        fn = support - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(
            {
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    status = (
        "INVALID"
        if incomplete
        else "DEGRADED"
        if errors or truncated or parse_failures
        else "VALID"
    )
    summary = {
        "status": status,
        "rows": len(rows),
        "expected_rows": expected,
        "incomplete": incomplete,
        "error_rows": errors,
        "truncated_rows": truncated,
        "parse_failure_rows": parse_failures,
        "accuracy": correct / max(len(rows), 1),
        "macro_f1_present_classes": (
            sum(item["f1"] for item in per_class) / max(len(per_class), 1)
        ),
        "present_classes": len(per_class),
    }
    by_group = {}
    for group in sorted({str(row.get("ref_group", "all")) for row in rows}):
        members = [row for row in rows if str(row.get("ref_group", "all")) == group]
        by_group[group] = {
            "rows": len(members),
            "accuracy": sum(
                normalize_class_name(row.get("gt_class"))
                == normalize_class_name(row.get("pred_class"))
                for row in members
            )
            / max(len(members), 1),
        }
    summary["by_ref_group"] = by_group
    output = (
        Path(args.output)
        if args.output
        else pred_path.with_name(pred_path.stem + "_metrics.json")
    )
    write_json(
        output,
        {
            "summary": summary,
            "per_class": per_class,
            "confusion": [
                {"gt": gt, "pred": pred, "count": count}
                for (gt, pred), count in sorted(confusion.items())
            ],
            "manifest": manifest,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if incomplete and settings["evaluation"].get("fail_on_incomplete", True):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
