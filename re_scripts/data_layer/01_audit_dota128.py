#!/usr/bin/env python3
"""Audit DOTA OBB data and separate fatal corruption from recoverable edge boxes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import (  # noqa: E402
    discover_images,
    inspect_obb_object,
    load_obb_label,
    load_settings,
    settings_from_cli,
    write_json,
)


def _print_summary(report: dict, out: Path) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    if report["passed"] and report.get("warnings"):
        status = "PASS_WITH_WARNINGS"
    print(f"[DOTA AUDIT] {status}")
    for split, item in report["splits"].items():
        print(
            f"  {split}: images={item['images']} raw={item['objects_raw']} "
            f"valid={item['objects_valid']} invalid={item['invalid_objects']} "
            f"invalid_rate={item['invalid_rate']:.2%}"
        )
        if item.get("invalid_reasons"):
            reasons = ", ".join(f"{k}={v}" for k, v in item["invalid_reasons"].items())
            print(f"    reasons: {reasons}")
    if report.get("val_only_classes"):
        print("  val-only classes: " + ", ".join(report["val_only_classes"]))
    if report.get("fatal_errors"):
        print("  fatal errors:")
        for x in report["fatal_errors"]:
            print(f"    - {x}")
    print(f"  policy: invalid_object_policy={report['invalid_object_policy']}")
    print(f"  report: {out}")
    print(f"  next: {report['next_step']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--verbose", action="store_true", help="打印完整 JSON 报告")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    root = Path(settings["paths"]["dota128_root"])
    quality = settings.get("data_quality", {})
    audit_mode = str(quality.get("audit_mode", "warn")).lower()
    min_area = float(quality.get("min_polygon_area", 1.0))
    allow_val_only = bool(quality.get("allow_val_only_classes", True))
    fail_on_overlap = bool(quality.get("fail_on_train_val_overlap", True))

    report = {
        "dataset_root": str(root),
        "audit_mode": audit_mode,
        "invalid_object_policy": quality.get("invalid_object_policy", "drop"),
        "splits": {},
        "fatal_errors": [],
        "warnings": [],
    }
    stems: dict[str, set[str]] = {}

    if not root.is_dir():
        report["fatal_errors"].append(f"dataset root does not exist: {root}")

    for split in settings["data_conversion"].get("splits", ["train", "val"]):
        try:
            rows = discover_images(root, split)
        except Exception as exc:
            report["fatal_errors"].append(f"{split}: {type(exc).__name__}: {exc}")
            rows = []
        classes: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        invalid_reasons: Counter[str] = Counter()
        objects = 0
        invalid_objects = 0
        split_stems: set[str] = set()
        examples: list[dict] = []

        if not rows:
            report["fatal_errors"].append(f"{split}: no image/label pairs found")

        for image_path, label_path in rows:
            split_stems.add(image_path.stem)
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                obs, mode = load_obb_label(label_path, width, height)
                modes[mode] += 1
                for obj in obs:
                    classes[obj["class_name"]] += 1
                    objects += 1
                    status = inspect_obb_object(obj, width, height, min_area)
                    if not status["valid"]:
                        invalid_objects += 1
                        invalid_reasons.update(status["reasons"])
                        if len(examples) < 20:
                            examples.append(
                                {
                                    "image": str(image_path),
                                    "label": str(label_path),
                                    "object_index": obj["object_index"],
                                    "class_name": obj["class_name"],
                                    "reasons": status["reasons"],
                                    "points_pixel": obj["points_pixel"],
                                }
                            )
            except Exception as exc:
                report["fatal_errors"].append(
                    f"{split}:{image_path.name}: {type(exc).__name__}: {exc}"
                )

        stems[split] = split_stems
        report["splits"][split] = {
            "images": len(rows),
            "objects_raw": objects,
            "objects_valid": objects - invalid_objects,
            "invalid_objects": invalid_objects,
            "invalid_rate": round(invalid_objects / max(1, objects), 6),
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
            "invalid_examples": examples,
            "class_counts": dict(sorted(classes.items())),
            "coordinate_modes": dict(sorted(modes.items())),
        }
        if invalid_objects:
            report["warnings"].append(
                f"{split}: {invalid_objects}/{objects} invalid or out-of-image OBBs; "
                f"downstream policy={quality.get('invalid_object_policy', 'drop')}"
            )

    if "train" in stems and "val" in stems:
        overlap = sorted(stems["train"] & stems["val"])
        report["train_val_stem_overlap"] = overlap
        train_cls = set(report["splits"].get("train", {}).get("class_counts", {}))
        val_cls = set(report["splits"].get("val", {}).get("class_counts", {}))
        report["class_overlap"] = sorted(train_cls & val_cls)
        report["val_only_classes"] = sorted(val_cls - train_cls)
        if overlap and fail_on_overlap:
            report["fatal_errors"].append(
                f"train/val image stem overlap: {len(overlap)} images"
            )
        if report["val_only_classes"]:
            message = "validation-only classes: " + ", ".join(report["val_only_classes"])
            if allow_val_only:
                report["warnings"].append(message)
            else:
                report["fatal_errors"].append(message)

    report["pipeline_passed"] = not report["fatal_errors"]
    report["strict_passed"] = report["pipeline_passed"] and all(
        item.get("invalid_objects", 0) == 0 for item in report["splits"].values()
    )
    report["passed"] = report["strict_passed"] if audit_mode == "strict" else report["pipeline_passed"]
    report["next_step"] = (
        "python data_layer/02_build_dota128_ref.py"
        if report["passed"]
        else "fix fatal_errors, then rerun audit"
    )

    out = Path(settings["paths"]["pipeline_work_root"]) / "reports" / "dota128_audit.json"
    write_json(out, report)
    _print_summary(report, out)
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
