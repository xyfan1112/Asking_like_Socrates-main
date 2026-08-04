#!/usr/bin/env python3
"""Build SocraticAgent inputs and non-contradictory Direct OBB SFT data."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main_layer"))
from common import (  # noqa: E402
    CUSTOM_TAXONOMY,
    DOTA_CLASSES,
    QA_LANGUAGE,
    TAXONOMY_SHA256,
    coordinate_instruction,
    coordinate_points,
    format_flat_points,
    load_settings,
    read_jsonl,
    settings_from_cli,
    split_center_rois,
    stable_id,
    write_json,
    write_jsonl,
)
from data_layer.qa_i18n import direct_detection_question  # noqa: E402


def _answer_for(row: dict[str, Any], target: str) -> str:
    points = {
        "pixel_obb": row["obb_pixel"],
        "norm100_obb": row["obb_norm100"],
        "norm1000_obb": row["obb_norm1000"],
    }[target]
    return f"{row['class_name']}|{format_flat_points(points, 0)}"


def _stratified_round_robin(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Interleave task/class buckets so a debug prefix spans images and classes."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], deque] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row.get("class_name", ""))].append(row)
    for key, items in grouped.items():
        rng.shuffle(items)
        buckets[key] = deque(items)
    keys = list(buckets)
    rng.shuffle(keys)
    output: list[dict[str, Any]] = []
    while keys:
        next_keys = []
        for key in keys:
            if buckets[key]:
                output.append(buckets[key].popleft())
            if buckets[key]:
                next_keys.append(key)
        rng.shuffle(next_keys)
        keys = next_keys
    return output


def _spatial_chunks(
    items: list[dict[str, Any]],
    max_objects: int,
    width: int,
    height: int,
) -> list[tuple[list[float], list[dict[str, Any]]]]:
    return split_center_rois(items, max_objects, width, height)


def _roi_to_target(roi: list[float], width: int, height: int, target: str) -> list[float]:
    x1, y1, x2, y2 = roi
    if target == "pixel_obb":
        return [x1, y1, x2, y2]
    scale = 100.0 if target == "norm100_obb" else 1000.0
    return [
        x1 / max(width, 1) * scale,
        y1 / max(height, 1) * scale,
        x2 / max(width, 1) * scale,
        y2 / max(height, 1) * scale,
    ]


def _direct_samples(
    all_rows: list[dict[str, Any]],
    target: str,
    max_objects: int,
    negative_classes_per_image: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_image[row["image_path"]].append(row)
    samples = []
    rng = random.Random(seed)
    for image_path, image_rows in sorted(by_image.items()):
        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in image_rows:
            by_class[row["class_name"]].append(row)
        width, height = image_rows[0]["image_width"], image_rows[0]["image_height"]
        requested_classes: list[tuple[str, list[dict[str, Any]], bool]] = [
            (cls, cls_rows, False) for cls, cls_rows in sorted(by_class.items())
        ]
        absent = [name for name in DOTA_CLASSES if name not in by_class]
        rng.shuffle(absent)
        requested_classes.extend(
            (cls, [], True) for cls in sorted(absent[: max(0, negative_classes_per_image)])
        )
        for cls, cls_rows, is_negative in requested_classes:
            cls_rows.sort(key=lambda r: (r["center_pixel"][1], r["center_pixel"][0]))
            spatial_groups = (
                [([0.0, 0.0, float(width), float(height)], [])]
                if is_negative
                else _spatial_chunks(cls_rows, max_objects, width, height)
            )
            for chunk_idx, (roi_pixel, chunk) in enumerate(spatial_groups):
                lines = []
                for row in chunk:
                    pts = {
                        "pixel_obb": row["obb_pixel"],
                        "norm100_obb": row["obb_norm100"],
                        "norm1000_obb": row["obb_norm1000"],
                    }[target]
                    lines.append(f"{cls}|1.0|{format_flat_points(pts, 0)}")
                answer = "FINAL_DETECTIONS\n" + "\n".join(lines) + "\nEND_DETECTIONS"
                roi_target = _roi_to_target(roi_pixel, width, height, target)
                roi_text = ",".join(f"{value:.2f}".rstrip("0").rstrip(".") for value in roi_target)
                question = direct_detection_question(
                    width=width,
                    height=height,
                    class_name=cls,
                    roi_text=roi_text,
                    coordinate_target=target,
                    lang=QA_LANGUAGE,
                )
                direct_salt = (
                    "direct_roi_v4_3_3"
                    if not CUSTOM_TAXONOMY and QA_LANGUAGE == "en"
                    else f"direct_roi_v4_3_3|{QA_LANGUAGE}|{TAXONOMY_SHA256}"
                )
                sample_id = stable_id(
                    image_path,
                    cls,
                    roi_text,
                    target,
                    direct_salt,
                )
                samples.append(
                    {
                        "messages": [
                            {"role": "user", "content": f"<image>\n{question}"},
                            {"role": "assistant", "content": "</think>\n" + answer},
                        ],
                        "images": [image_path],
                        "metadata": {
                            "task": "detection_obb_by_class",
                            "sample_id": sample_id,
                            "qa_language": QA_LANGUAGE,
                            "taxonomy_mode": "custom" if CUSTOM_TAXONOMY else "dota",
                            "taxonomy_sha256": TAXONOMY_SHA256,
                            "class_name": cls,
                            "coordinate_target": target,
                            "chunk_index": chunk_idx,
                            "objects_in_chunk": len(chunk),
                            "negative_sample": bool(is_negative),
                            "roi_pixel": [round(value, 4) for value in roi_pixel],
                            "roi_target": [round(value, 4) for value in roi_target],
                            "source_ref_ids": [row["id"] for row in chunk],
                            "image_width": width,
                            "image_height": height,
                        },
                    }
                )
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--force", action="store_true", help="忽略 validation_report 门控，不建议正式实验使用")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    seed = int(settings["project"]["seed"])
    cfg = settings["data_conversion"]
    target = cfg.get("coordinate_target", "norm1000_obb")
    ref_root = Path(settings["paths"]["dota128_ref_root"])
    work = Path(settings["paths"]["pipeline_work_root"]) / "agent_inputs"
    work.mkdir(parents=True, exist_ok=True)
    validation_path = ref_root / "validation_report.json"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"缺少 {validation_path}。先运行: python data_layer/03_validate_dota128_ref.py"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passed", False) and not args.force:
        raise RuntimeError(
            "DOTA-Ref 未通过验证，拒绝生成 agent_inputs。"
            "先查看 validation_report.json 并重新运行 02/03；调试时才可使用 --force。"
        )
    report: dict[str, Any] = {
        "schema_version": "agent_inputs_v4_3_3",
        "coordinate_target": target,
        "qa_language": QA_LANGUAGE,
        "taxonomy_mode": "custom" if CUSTOM_TAXONOMY else "dota",
        "taxonomy_sha256": TAXONOMY_SHA256,
        "num_classes": len(DOTA_CLASSES),
        "validation_report": str(validation_path),
        "validation_passed": bool(validation.get("passed", False)),
        "forced": bool(args.force),
        "splits": {},
    }

    for split in cfg.get("splits", ["train", "val"]):
        refs = read_jsonl(ref_root / f"{split}.jsonl")
        all_rows = read_jsonl(ref_root / f"{split}_all.jsonl")
        source_refs = (
            [row for row in refs if row.get("socratic_eligible")]
            if split == "train" and cfg.get("train_agent_use_socratic_subset", True)
            else refs
        )
        agent_rows: list[dict[str, Any]] = []

        by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source_refs:
            by_image[row["image_path"]].append(row)
        rng = random.Random(seed + (0 if split == "train" else 10000))

        for image_path, image_refs in sorted(by_image.items()):
            image_refs = list(image_refs)
            rng.shuffle(image_refs)
            image_refs = image_refs[: int(cfg.get("max_objects_per_image_for_agent", 24))]
            for row in image_refs:
                base = {
                    "image_path": image_path,
                    "image_width": row["image_width"],
                    "image_height": row["image_height"],
                    "class_name": row["class_name"],
                    "class_id": row["class_id"],
                    "object_index": row["object_index"],
                    "source_ref_id": row["id"],
                    "scene_id": row.get("scene_id"),
                    "coordinate_target": target,
                    "qa_language": row.get("qa_language", QA_LANGUAGE),
                    "lang": row.get("qa_language", QA_LANGUAGE),
                    "taxonomy_mode": row.get(
                        "taxonomy_mode",
                        "custom" if CUSTOM_TAXONOMY else "dota",
                    ),
                    "taxonomy_sha256": row.get(
                        "taxonomy_sha256", TAXONOMY_SHA256
                    ),
                    "obb_pixel": row["obb_pixel"],
                    "obb_norm100": row["obb_norm100"],
                    "obb_norm1000": row["obb_norm1000"],
                    "hbb_norm1000": row["hbb_norm1000"],
                    "classification_focus_hbb_norm1000": row.get(
                        "classification_focus_hbb_norm1000"
                    ),
                }
                if cfg.get("build_grounding_tasks", True):
                    agent_rows.append(
                        {
                            **base,
                            "id": stable_id(row["id"], "grounding_v4_3_3"),
                            "task": "ref_grounding_obb",
                            "query": f"<image>\n{row['grounding_question']}",
                            "gt": _answer_for(row, target),
                            "reference": row["reference_grounding"],
                            "reference_type": "grounding",
                        }
                    )
                if cfg.get("build_classification_tasks", True):
                    agent_rows.append(
                        {
                            **base,
                            "id": stable_id(row["id"], "classification_v4_3_3"),
                            "task": "ref_classification",
                            "query": f"<image>\n{row['classification_question']}",
                            "gt": row["class_name"],
                            "reference": row["reference_classification"],
                            "reference_type": "classification_masked_label",
                        }
                    )

        if cfg.get("agent_input_shuffle") == "stratified_round_robin":
            agent_rows = _stratified_round_robin(agent_rows, seed + (1 if split == "train" else 10001))
        else:
            rng.shuffle(agent_rows)

        agent_path = work / f"{split}_agent_inputs.jsonl"
        direct_path = work / f"{split}_direct.json"
        write_jsonl(agent_path, agent_rows)
        direct = _direct_samples(
            all_rows,
            target,
            int(cfg.get("max_objects_per_direct_sample", 40)),
            int(cfg.get("direct_negative_classes_per_image", 2)),
            seed + (0 if split == "train" else 10000),
        ) if cfg.get("build_detection_tasks", True) else []
        direct_path.write_text(json.dumps(direct, ensure_ascii=False, indent=2), encoding="utf-8")

        task_counts = defaultdict(int)
        class_counts = defaultdict(int)
        image_counts = set()
        for row in agent_rows:
            task_counts[row["task"]] += 1
            class_counts[row["class_name"]] += 1
            image_counts.add(row["image_path"])
        report["splits"][split] = {
            "ref_rows": len(refs),
            "socratic_source_rows": len(source_refs),
            "all_objects": len(all_rows),
            "agent_inputs": len(agent_rows),
            "agent_images": len(image_counts),
            "agent_task_counts": dict(sorted(task_counts.items())),
            "agent_class_counts": dict(sorted(class_counts.items())),
            "direct_samples": len(direct),
            "agent_file": str(agent_path),
            "direct_file": str(direct_path),
        }

    report["passed"] = all(
        item.get("agent_inputs", 0) > 0 and item.get("direct_samples", 0) > 0
        for item in report["splits"].values()
    )
    report["next_step"] = (
        "python main_layer/run.py build-official-parquet --settings settings.json --split train"
        if report["passed"]
        else "inspect agent_inputs/build_report.json"
    )
    report_path = work / "build_report.json"
    write_json(report_path, report)
    print(f"[AGENT INPUTS] {'PASS' if report['passed'] else 'FAIL'}")
    for split, item in report["splits"].items():
        print(
            f"  {split}: refs={item['ref_rows']} agent_inputs={item['agent_inputs']} "
            f"socratic_source={item['socratic_source_rows']} "
            f"direct_samples={item['direct_samples']} images={item['agent_images']}"
        )
        print(f"    agent: {item['agent_file']}")
        print(f"    direct: {item['direct_file']}")
    print(f"  report: {report_path}")
    print(f"  next: {report['next_step']}")
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
