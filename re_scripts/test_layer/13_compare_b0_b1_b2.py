#!/usr/bin/env python3
"""Collect the three controlled experiment summaries into JSON/CSV/Markdown."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import load_settings, settings_from_cli, write_json  # noqa: E402


def summary(path: Path) -> dict:
    if not path.is_file():
        return {"_missing": str(path)}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("summary", value)


def get(data: dict, key: str):
    return None if data.get("_missing") else data.get(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument(
        "--model-keys",
        nargs="+",
        default=["rs_eot", "rs_eot_b1_direct", "rs_eot_b2_socratic"],
    )
    ap.add_argument("--labels", nargs="+", default=["B0", "B1", "B2"])
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    if len(args.labels) != len(args.model_keys):
        raise ValueError("--labels and --model-keys must have equal lengths")
    settings = load_settings(settings_from_cli(__file__, args.settings))
    root = Path(settings["paths"]["test_run_root"])
    rows = []
    missing = []
    for label, key in zip(args.labels, args.model_keys):
        detection_path = (
            root
            / "dota_detection"
            / key
            / "val_dota_class_conditioned_roi_v2_k1_metrics.json"
        )
        grounding_path = (
            root
            / "ref_grounding"
            / "dota_ref"
            / key
            / "val_ref_grounding_k1_metrics.json"
        )
        classification_path = (
            root / "ref_classification" / key / "val_k1_metrics.json"
        )
        detection = summary(detection_path)
        grounding = summary(grounding_path)
        classification = summary(classification_path)
        for value in (detection, grounding, classification):
            if value.get("_missing"):
                missing.append(value["_missing"])
        rows.append(
            {
                "experiment": label,
                "model_key": key,
                "det_status": get(detection, "status"),
                "det_format_success": get(
                    detection, "output_protocol_success_rate"
                ),
                "det_mean_iou": get(detection, "mean_iou_per_gt"),
                "det_recall_iou_0_5": get(detection, "recall_iou_0_5"),
                "det_recall_iou_0_7": get(detection, "recall_iou_0_7"),
                "det_negative_accuracy": get(
                    detection, "negative_query_accuracy"
                ),
                "det_map50": get(detection, "map50"),
                "ref_status": get(grounding, "status"),
                "ref_mean_iou": get(grounding, "mean_iou_all_runs"),
                "ref_loc_acc_0_5": get(
                    grounding, "localization_acc_iou_0_5"
                ),
                "ref_loc_acc_0_7": get(
                    grounding, "localization_acc_iou_0_7"
                ),
                "ref_joint_acc_0_5": get(
                    grounding, "joint_class_iou_acc_0_5"
                ),
                "ref_answer_class_accuracy": get(
                    grounding, "class_accuracy"
                ),
                "cls_status": get(classification, "status"),
                "cls_accuracy": get(classification, "accuracy"),
                "cls_macro_f1": get(
                    classification, "macro_f1_present_classes"
                ),
            }
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "comparison" / "b0_b1_b2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "comparison.json",
        {
            "schema_version": "b0_b1_b2_comparison_v1",
            "rows": rows,
            "missing": missing,
            "passed": not missing,
        },
    )
    columns = list(rows[0]) if rows else []
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "" if row[column] is None else f"{row[column]:.4f}"
                if isinstance(row[column], float)
                else str(row[column])
                for column in columns
            )
            + " |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[COMPARISON] {output_dir}")
    if missing:
        print("[MISSING]")
        for path in missing:
            print(f"  - {path}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
