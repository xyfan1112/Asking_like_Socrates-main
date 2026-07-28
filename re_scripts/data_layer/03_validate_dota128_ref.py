#!/usr/bin/env python3
"""Validate DOTA-Ref lineage, semantic uniqueness, coordinates, and isolation."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main_layer"))
from common import DOTA_CLASSES, discover_images, load_settings, polygon_area, read_jsonl, settings_from_cli, write_json  # noqa: E402
from data_layer.ref_semantics import resolve_reference_matches  # noqa: E402

UNNATURAL_CLASS_RE = re.compile(r"\b(?:at the (?:top|bottom|far left|far right) of its class|of its class)\b", re.I)
ORIENTATION_RE = re.compile(r"\b(?:roughly horizontal|roughly vertical|diagonally ascending to the right|diagonally descending to the right)\b", re.I)
NO_ORIENTATION_CLASSES = {"storage tank", "roundabout", "harbor"}

ORDINAL_RE = re.compile(r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", re.I)
COORD_RE = re.compile(r"(?:\[|\()\s*-?\d+(?:\.\d+)?\s*[, ]+\s*-?\d+(?:\.\d+)?")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
AREA_AREA_RE = re.compile(r"\barea\s+area\b", re.I)
EXPECTED_SCHEMA = "dota_ref_v4_3_3"
TARGET_ALIASES = {
    "plane": ("plane", "airplane", "aircraft", "jet"),
    "ship": ("ship", "boat", "vessel", "watercraft"),
    "storage tank": ("storage tank", "tank"),
    "baseball diamond": ("baseball diamond", "baseball field", "baseball"),
    "tennis court": ("tennis court", "tennis"),
    "basketball court": ("basketball court", "basketball"),
    "ground track field": ("ground track field", "running track", "track field"),
    "harbor": ("harbor", "port", "marina"),
    "bridge": ("bridge",),
    "large vehicle": ("large vehicle", "truck", "lorry", "trailer", "bus"),
    "small vehicle": ("small vehicle", "car", "sedan", "suv", "pickup"),
    "helicopter": ("helicopter", "rotorcraft"),
    "roundabout": ("roundabout", "traffic circle"),
    "soccer ball field": ("soccer ball field", "soccer field", "football field"),
    "swimming pool": ("swimming pool", "pool"),
}


def _points_in_pixel_bounds(points, width: int, height: int) -> bool:
    return (
        isinstance(points, list)
        and len(points) == 4
        and all(
            isinstance(p, list)
            and len(p) == 2
            and all(math.isfinite(float(v)) for v in p)
            and 0.0 <= float(p[0]) <= max(0.0, width - 1.0)
            and 0.0 <= float(p[1]) <= max(0.0, height - 1.0)
            for p in points
        )
    )


def _points_in_range(points, upper: float) -> bool:
    return (
        isinstance(points, list)
        and len(points) == 4
        and all(
            isinstance(p, list)
            and len(p) == 2
            and all(math.isfinite(float(v)) and 0.0 <= float(v) <= upper for v in p)
            for p in points
        )
    )


def _flat_in_range(values, expected: int, upper: float) -> bool:
    return (
        isinstance(values, list)
        and len(values) == expected
        and all(math.isfinite(float(v)) and 0.0 <= float(v) <= upper for v in values)
    )


def _focus_contains_target(row: dict) -> bool:
    focus = row.get("classification_focus_hbb_norm1000")
    center = row.get("center_pixel")
    width, height = float(row.get("image_width", 0)), float(row.get("image_height", 0))
    if not _flat_in_range(focus, 4, 1000.0) or not isinstance(center, list) or len(center) != 2:
        return False
    cx = float(center[0]) / max(width, 1.0) * 1000.0
    cy = float(center[1]) / max(height, 1.0) * 1000.0
    return float(focus[0]) <= cx <= float(focus[2]) and float(focus[1]) <= cy <= float(focus[3])


def _question_has_exact_focus(row: dict, field: str) -> bool:
    focus = row.get("classification_focus_hbb_norm1000")
    if not _flat_in_range(focus, 4, 1000.0):
        return False
    expected = "[" + ",".join(str(int(round(float(value)))) for value in focus) + "]"
    text = re.sub(r"\s+", " ", str(row.get(field) or "")).lower()
    return "focus region" in text and expected in text.replace(" ", "")


def _answer_valid(row: dict) -> bool:
    answer = str(row.get("grounding_answer") or row.get("answer") or "")
    if "|" not in answer:
        return False
    nums = [float(x) for x in NUMBER_RE.findall(answer.split("|", 1)[1])]
    target = row.get("coordinate_target")
    upper = {"pixel_obb": None, "norm100_obb": 100.0, "norm1000_obb": 1000.0}.get(target)
    if len(nums) != 8:
        return False
    if upper is None:
        width, height = int(row.get("image_width", 0)), int(row.get("image_height", 0))
        return all(
            0.0 <= nums[i] <= max(0.0, width - 1.0) if i % 2 == 0 else 0.0 <= nums[i] <= max(0.0, height - 1.0)
            for i in range(8)
        )
    return all(0.0 <= v <= upper for v in nums)


def _contains_target_alias(text: str, class_name: str) -> bool:
    low = str(text or "").lower()
    return any(
        re.search(rf"\b{re.escape(alias)}s?\b", low)
        for alias in TARGET_ALIASES.get(class_name, (class_name,))
        if alias
    )


def _semantic_descriptors(rows: list[dict]) -> list[dict]:
    return [
        {
            "object_index": row.get("object_index"),
            "class_name": row.get("class_name"),
            "center_pixel": row.get("center_pixel"),
            "semantic_region": row.get("reference_region"),
            "geometry_phrase_used": row.get("geometry_phrase_used"),
            "semantic_extremes": row.get("semantic_extremes") or [],
        }
        for row in rows
    ]


def _validate_rows(
    rows: list[dict],
    selected: bool,
    all_by_image: dict[str, list[dict]] | None = None,
) -> tuple[Counter, Counter]:
    stats = Counter()
    classes = Counter()
    per_image_ground = defaultdict(Counter)
    per_image_class = defaultdict(Counter)

    for row in rows:
        image = str(row.get("image_path", ""))
        width, height = int(row.get("image_width", 0)), int(row.get("image_height", 0))
        canonical = str(row.get("class_name", "")).lower()
        classes[canonical or "unknown"] += 1
        if row.get("schema_version") != EXPECTED_SCHEMA:
            stats["schema_version_mismatch"] += 1

        if not Path(image).is_file():
            stats["missing_images"] += 1
        points = row.get("obb_pixel") or row.get("obj_corner_pixel")
        if not isinstance(points, list) or len(points) != 4 or polygon_area(points) <= 1:
            stats["invalid_polygon"] += 1
        if not _points_in_pixel_bounds(points, width, height):
            stats["pixel_out_of_bounds"] += 1
        if not _points_in_range(row.get("obb_norm100"), 100.0):
            stats["norm100_out_of_range"] += 1
        if not _points_in_range(row.get("obb_norm1000"), 1000.0):
            stats["norm1000_out_of_range"] += 1
        if not _flat_in_range(row.get("hbb_norm1000"), 4, 1000.0):
            stats["hbb_norm1000_out_of_range"] += 1

        if selected:
            ground = str(row.get("reference_grounding", ""))
            class_ref = str(row.get("reference_classification", ""))
            per_image_ground[image][ground] += 1
            focus_key = tuple(round(float(v), 3) for v in row.get("classification_focus_hbb_norm1000", []))
            per_image_class[image][(class_ref, focus_key)] += 1
            if not ground or not class_ref:
                stats["empty_references"] += 1
            if COORD_RE.search(ground) or COORD_RE.search(class_ref):
                stats["coordinate_leak"] += 1
            if ORDINAL_RE.search(ground) or ORDINAL_RE.search(class_ref):
                stats["ordinal_reference"] += 1
            if UNNATURAL_CLASS_RE.search(ground) or UNNATURAL_CLASS_RE.search(class_ref):
                stats["unnatural_class_relative_phrase"] += 1
            if AREA_AREA_RE.search(ground) or AREA_AREA_RE.search(class_ref):
                stats["duplicated_area_word"] += 1
            if canonical in NO_ORIENTATION_CLASSES and (ORIENTATION_RE.search(ground) or ORIENTATION_RE.search(class_ref)):
                stats["unstable_orientation_phrase"] += 1
            if canonical and re.search(rf"\b{re.escape(canonical)}\b", class_ref.lower()):
                stats["classification_label_leak"] += 1
            if canonical and _contains_target_alias(class_ref, canonical):
                stats["classification_alias_leak"] += 1
            semantics = (
                row.get("reference_semantics")
                if isinstance(row.get("reference_semantics"), dict)
                else {}
            )
            constraints = (
                semantics.get("constraints")
                if isinstance(semantics.get("constraints"), dict)
                else {}
            )
            if not constraints:
                stats["missing_semantic_reference_metadata"] += 1
            else:
                image_rows = (all_by_image or {}).get(image, [])
                recomputed = resolve_reference_matches(
                    constraints,
                    _semantic_descriptors(image_rows),
                    width,
                    height,
                )
                expected = [int(row.get("object_index"))]
                stored = sorted(
                    int(value)
                    for value in (semantics.get("match_object_indices") or [])
                )
                if recomputed != expected:
                    stats["semantic_reference_not_unique"] += 1
                if stored != recomputed or not bool(semantics.get("semantic_unique")):
                    stats["semantic_resolution_mismatch"] += 1
            if not _answer_valid(row):
                stats["grounding_answer_invalid"] += 1
            if row.get("coordinate_target") not in {"pixel_obb", "norm100_obb", "norm1000_obb"}:
                stats["unknown_coordinate_target"] += 1
            if not _flat_in_range(row.get("classification_focus_hbb_norm1000"), 4, 1000.0):
                stats["classification_focus_out_of_range"] += 1
            elif not _focus_contains_target(row):
                stats["classification_focus_misses_target"] += 1
            if not _question_has_exact_focus(row, "grounding_question"):
                stats["grounding_question_focus_missing_or_mismatched"] += 1
            if not _question_has_exact_focus(row, "classification_question"):
                stats["classification_question_focus_missing_or_mismatched"] += 1
            if not isinstance(row.get("socratic_eligible"), bool):
                stats["missing_socratic_eligibility"] += 1
            if row.get("socratic_eligible") and row.get("socratic_reject_reasons"):
                stats["inconsistent_socratic_eligibility"] += 1

    if selected:
        stats["duplicate_grounding_reference_within_image"] = sum(
            v - 1 for c in per_image_ground.values() for v in c.values() if v > 1
        )
        stats["duplicate_classification_reference_within_image"] = sum(
            v - 1 for c in per_image_class.values() for v in c.values() if v > 1
        )
    return stats, classes


def _print_summary(report: dict, report_path: Path) -> None:
    print(f"[DOTA-REF VALIDATE] {'PASS' if report['passed'] else 'FAIL'}")
    for split, item in report["splits"].items():
        selected = item["selected"]
        all_stats = item["all"]
        print(
            f"  {split}: valid_objects={item['all_objects']} selected_refs={item['selected_references']} "
            f"pixel_oob(all/ref)={all_stats.get('pixel_out_of_bounds', 0)}/{selected.get('pixel_out_of_bounds', 0)} "
            f"answer_invalid={selected.get('grounding_answer_invalid', 0)}"
        )
        print(
            f"    duplicate_refs={selected.get('duplicate_grounding_reference_within_image', 0)} "
            f"class_leak={selected.get('classification_label_leak', 0)} "
            f"alias_leak={selected.get('classification_alias_leak', 0)} "
            f"semantic_non_unique={selected.get('semantic_reference_not_unique', 0)} "
            f"ordinal={selected.get('ordinal_reference', 0)} "
            f"unnatural={selected.get('unnatural_class_relative_phrase', 0)} "
            f"unstable_orientation={selected.get('unstable_orientation_phrase', 0)}"
        )
    if report.get("critical"):
        print("  critical:")
        for item in report["critical"]:
            print(f"    - {item}")
    if report.get("warnings"):
        print("  warnings:")
        for item in report["warnings"]:
            print(f"    - {item}")
    print(f"  report: {report_path}")
    print(f"  next: {report['next_step']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--verbose", action="store_true", help="打印完整 JSON 报告")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    root = Path(settings["paths"]["dota128_ref_root"])
    report = {"schema_version": "dota_ref_v4_3_3", "splits": {}, "critical": [], "warnings": []}
    split_scene_ids: dict[str, set[str]] = {}
    split_object_classes: dict[str, set[str]] = {}

    for split in settings["data_conversion"].get("splits", ["train", "val"]):
        selected_path = root / f"{split}.jsonl"
        all_path = root / f"{split}_all.jsonl"
        if not selected_path.is_file() or not all_path.is_file():
            report["critical"].append(f"{split}: missing {selected_path.name} or {all_path.name}")
            continue
        selected = read_jsonl(selected_path)
        all_rows = read_jsonl(all_path)
        all_by_image: dict[str, list[dict]] = defaultdict(list)
        for row in all_rows:
            all_by_image[str(row.get("image_path", ""))].append(row)
        selected_stats, classes = _validate_rows(
            selected,
            selected=True,
            all_by_image=all_by_image,
        )
        all_stats, all_classes = _validate_rows(all_rows, selected=False)
        split_object_classes[split] = set(all_classes)

        all_ids = {x.get("id") for x in all_rows}
        selected_ids = {x.get("id") for x in selected}
        selected_not_in_all = len(selected_ids - all_ids)
        if selected_not_in_all:
            selected_stats["selected_not_in_all"] = selected_not_in_all
        split_scene_ids[split] = {
            image_path.stem.split("__", 1)[0]
            for image_path, _ in discover_images(
                Path(settings["paths"]["dota128_root"]), split
            )
        }

        socratic_path = root / f"{split}_socratic.jsonl"
        socratic_rows = read_jsonl(socratic_path) if socratic_path.is_file() else []
        expected_socratic_ids = {row.get("id") for row in selected if row.get("socratic_eligible")}
        actual_socratic_ids = {row.get("id") for row in socratic_rows}
        if expected_socratic_ids != actual_socratic_ids:
            selected_stats["socratic_subset_mismatch"] = len(
                expected_socratic_ids.symmetric_difference(actual_socratic_ids)
            )

        report["splits"][split] = {
            "all_objects": len(all_rows),
            "selected_references": len(selected),
            "socratic_references": len(socratic_rows),
            "scene_ids": len(split_scene_ids[split]),
            "classes": dict(sorted(classes.items())),
            "all": dict(sorted(all_stats.items())),
            "selected": dict(sorted(selected_stats.items())),
        }

        fatal_keys = {
            "missing_images",
            "invalid_polygon",
            "pixel_out_of_bounds",
            "norm100_out_of_range",
            "norm1000_out_of_range",
            "hbb_norm1000_out_of_range",
            "schema_version_mismatch",
        }
        for key in fatal_keys:
            value = all_stats.get(key, 0)
            if value:
                report["critical"].append(f"{split}:all:{key}={value}")
        selected_fatal = fatal_keys | {
            "empty_references",
            "coordinate_leak",
            "classification_label_leak",
            "classification_alias_leak",
            "missing_semantic_reference_metadata",
            "semantic_reference_not_unique",
            "semantic_resolution_mismatch",
            "grounding_answer_invalid",
            "duplicate_grounding_reference_within_image",
            "duplicate_classification_reference_within_image",
            "selected_not_in_all",
            "unknown_coordinate_target",
            "unnatural_class_relative_phrase",
            "unstable_orientation_phrase",
            "duplicated_area_word",
            "classification_focus_out_of_range",
            "classification_focus_misses_target",
            "grounding_question_focus_missing_or_mismatched",
            "classification_question_focus_missing_or_mismatched",
            "missing_socratic_eligibility",
            "inconsistent_socratic_eligibility",
            "socratic_subset_mismatch",
        }
        for key in selected_fatal:
            value = selected_stats.get(key, 0)
            if value:
                report["critical"].append(f"{split}:ref:{key}={value}")
        if selected_stats.get("ordinal_reference", 0):
            report["warnings"].append(f"{split}:ordinal_reference={selected_stats['ordinal_reference']}")
        missing_classes = sorted(set(DOTA_CLASSES) - set(classes))
        if missing_classes:
            report["warnings"].append(f"{split}:ref missing classes {missing_classes}; Direct data keeps valid boxes")

    split_names = sorted(split_scene_ids)
    require_scene_disjoint = bool(
        settings.get("data_quality", {}).get("require_scene_disjoint_splits", True)
    )
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlap = sorted(split_scene_ids[left] & split_scene_ids[right])
            if overlap:
                message = f"scene_leakage:{left}<->{right}:{len(overlap)}:{overlap[:20]}"
                if require_scene_disjoint:
                    report["critical"].append(message)
                else:
                    report["warnings"].append(message)

    if "train" in split_object_classes and "val" in split_object_classes:
        val_only = sorted(split_object_classes["val"] - split_object_classes["train"])
        if val_only:
            message = f"val_only_classes:{val_only}"
            if settings.get("data_quality", {}).get("allow_val_only_classes", True):
                report["warnings"].append(message + "; these remain zero-shot classes")
            else:
                report["critical"].append(message)

    report["passed"] = not report["critical"]
    report["next_step"] = (
        "python data_layer/04_visualize_dota128_ref.py"
        if report["passed"]
        else "rerun 02_build_dota128_ref.py with data_quality.invalid_object_policy=drop, then validate again"
    )
    report_path = root / "validation_report.json"
    write_json(report_path, report)
    _print_summary(report, report_path)
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
