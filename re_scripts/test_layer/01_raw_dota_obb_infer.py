#!/usr/bin/env python3
"""DOTA OBB inference.

Primary protocol is class-conditioned detection, matching the Direct SFT task.
The all-class one-shot mode is retained only as a dense-output stress test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
sys.path.insert(0, str(ROOT / "test_layer"))
from common import (  # noqa: E402
    DOTA_CLASSES,
    append_jsonl,
    discover_images,
    image_data_url,
    load_settings,
    parse_obb_output_status,
    read_jsonl,
    settings_from_cli,
    split_center_rois,
)
from eval_common import chat_once, check_server, protocol, write_manifest  # noqa: E402


def coord_mode(name: str) -> str:
    return name or "normalized_0_1000"


def _roi_in_mode(roi: list[float], width: int, height: int, mode: str) -> list[float]:
    if mode == "pixel":
        return roi
    scale = 1000.0 if mode == "normalized_0_1000" else 100.0
    return [
        roi[0] / max(width, 1) * scale,
        roi[1] / max(height, 1) * scale,
        roi[2] / max(width, 1) * scale,
        roi[3] / max(height, 1) * scale,
    ]


def class_question(
    class_name: str,
    width: int,
    height: int,
    mode: str,
    roi_pixel: list[float],
) -> str:
    convention = "normalized coordinates in [0,1000] relative to the original image" if mode == "normalized_0_1000" else "original-image pixel coordinates"
    roi = _roi_in_mode(roi_pixel, width, height, mode)
    roi_text = ",".join(f"{value:.2f}".rstrip("0").rstrip(".") for value in roi)
    return (
        f"Image size: {width}x{height}. Detect every {class_name} object whose center lies inside "
        f"ROI [{roi_text}]. Use {convention}; answer coordinates remain relative to the full image. "
        "Output one class_name|confidence|x1,y1,x2,y2,x3,y3,x4,y4 line per object, "
        "with four corners in clockwise order, between FINAL_DETECTIONS and END_DETECTIONS. "
        "If none exists, output an empty FINAL_DETECTIONS block. Keep any reasoning brief "
        "so the complete final block is always produced."
    )


def all_class_question(width: int, height: int, mode: str) -> str:
    convention = "normalized coordinates in [0,1000] relative to the original image" if mode == "normalized_0_1000" else "original-image pixel coordinates"
    return (
        f"Image size: {width}x{height}. Detect all DOTA objects. Use only these classes: {', '.join(DOTA_CLASSES)}. "
        f"Use {convention}. Output one class_name|confidence|x1,y1,x2,y2,x3,y3,x4,y4 line per object "
        "between FINAL_DETECTIONS and END_DETECTIONS."
    )


def load_canonical_gt(settings: dict, split: str) -> list[dict[str, Any]]:
    """Load the exact sanitized OBB population used by Ref/Direct training."""
    path = Path(settings["paths"]["dota128_ref_root"]) / f"{split}_all.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical GT missing: {path}. Run build-ref and validate-ref before evaluation."
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        grouped.setdefault(str(Path(row["image_path"]).resolve()), []).append(row)
    records = []
    for image_path, _ in discover_images(
        Path(settings["paths"]["dota128_root"]), split
    ):
        canonical_rows = grouped.get(str(image_path.resolve()), [])
        if canonical_rows:
            width = int(canonical_rows[0]["image_width"])
            height = int(canonical_rows[0]["image_height"])
            label_mode = str(
                canonical_rows[0].get("label_coordinate_mode", "canonical")
            )
        else:
            with Image.open(image_path) as image:
                width, height = image.size
            label_mode = "canonical_empty_after_sanitize"
        records.append(
            {
                "image_path": image_path,
                "rows": canonical_rows,
                "width": width,
                "height": height,
                "label_mode": label_mode,
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--model-key", default="rs_eot")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--protocol", default="dota_class_conditioned", choices=["dota_class_conditioned", "dota_all_class_stress"])
    ap.add_argument("--class-scope", default="all", choices=["all", "present"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--output")
    ap.add_argument("--fresh", action="store_true", help="Delete this protocol's old result before inference.")
    args = ap.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    model = settings["models"][args.model_key]
    cfg = protocol(settings, args.protocol)
    k = args.k or int(cfg.get("k", 1))
    check_server(args.base_url, model["served_name"])
    image_rows = load_canonical_gt(settings, args.split)
    if args.max_images:
        image_rows = image_rows[: args.max_images]
    protocol_tag = f"{args.protocol}_roi_v2" if args.protocol == "dota_class_conditioned" else args.protocol
    output = Path(args.output) if args.output else Path(settings["paths"]["test_run_root"]) / "dota_detection" / args.model_key / f"{args.split}_{protocol_tag}_k{k}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        output.unlink(missing_ok=True)
        output.with_suffix(".manifest.json").unlink(missing_ok=True)
    done = set()
    if output.exists():
        done = {(r.get("query_id"), int(r.get("run_id", 0))) for r in read_jsonl(output, skip_bad=True)}

    expected = 0
    max_objects = int(
        settings["evaluation"].get(
            "dota_max_objects_per_roi",
            settings["data_conversion"].get("max_objects_per_direct_sample", 40),
        )
    )
    for image_index, image_record in enumerate(image_rows, 1):
        image_path = Path(image_record["image_path"])
        canonical_rows = image_record["rows"]
        width = int(image_record["width"])
        height = int(image_record["height"])
        label_mode = str(image_record["label_mode"])
        by_class = {name: [] for name in DOTA_CLASSES}
        for obj in canonical_rows:
            by_class[obj["class_name"]].append({
                "class_id": obj["class_id"],
                "class_name": obj["class_name"],
                "points": obj["obb_pixel"],
                "center_pixel": obj["center_pixel"],
                "object_index": obj["object_index"],
            })
        if args.protocol == "dota_class_conditioned":
            classes = DOTA_CLASSES if args.class_scope == "all" else [x for x in DOTA_CLASSES if by_class[x]]
            tasks = []
            for cls in classes:
                groups = (
                    split_center_rois(by_class[cls], max_objects, width, height)
                    if by_class[cls]
                    else [([0.0, 0.0, float(width), float(height)], [])]
                )
                for roi_index, (roi_pixel, gt_group) in enumerate(groups):
                    query_id = f"{args.split}/{image_path.name}/{cls}/roi{roi_index}"
                    tasks.append(
                        (
                            query_id,
                            cls,
                            class_question(
                                cls,
                                width,
                                height,
                                coord_mode(cfg.get("coordinate_mode")),
                                roi_pixel,
                            ),
                            gt_group,
                            roi_pixel,
                        )
                    )
        else:
            all_gt = [x for xs in by_class.values() for x in xs]
            tasks = [(
                f"{args.split}/{image_path.name}/ALL",
                None,
                all_class_question(width, height, coord_mode(cfg.get("coordinate_mode"))),
                all_gt,
                [0.0, 0.0, float(width), float(height)],
            )]
        expected += len(tasks) * k

        for query_id, requested_class, question, gt_objects, roi_pixel in tasks:
            for run_id in range(k):
                if (query_id, run_id) in done:
                    continue
                response = chat_once(
                    args.base_url,
                    model["served_name"],
                    [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                        {"type": "text", "text": question},
                    ]}],
                    temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]), max_tokens=int(cfg["max_tokens"]), retries=1,
                )
                raw = response["raw_output"]
                parsed = parse_obb_output_status(
                    raw,
                    width,
                    height,
                    coord_mode(cfg.get("coordinate_mode")),
                    strict=True,
                    require_markers=True,
                    expected_class=requested_class,
                    require_confidence=True,
                )
                predictions = parsed["predictions"]
                row = {
                    "query_id": query_id, "image_key": f"{args.split}/{image_path.name}", "image_path": str(image_path),
                    "requested_class": requested_class, "run_id": run_id, "model_key": args.model_key,
                    "protocol": args.protocol, "width": width, "height": height, "label_coordinate_mode": label_mode,
                    "ground_truth": gt_objects,
                    "predictions": predictions,
                    "roi_pixel": roi_pixel,
                    "canonical_gt": True,
                    "format_ok": parsed["format_ok"],
                    "parse_ok": parsed["parse_ok"],
                    "parse_reason": parsed["reason"],
                    "question": question,
                    **response,
                }
                append_jsonl(output, row)
                print(f"[{image_index}/{len(image_rows)}] class={requested_class or 'ALL'} run={run_id} gt={len(gt_objects)} pred={len(predictions)} error={bool(response['error'])}")

    manifest = {
        "protocol": args.protocol, "model_key": args.model_key, "split": args.split, "k": k,
        "class_scope": args.class_scope, "images": len(image_rows), "expected_rows": expected,
        "output": str(output), "settings_protocol": cfg,
        "schema_version": "dota_detection_roi_v2",
        "canonical_gt_source": str(Path(settings["paths"]["dota128_ref_root"]) / f"{args.split}_all.jsonl"),
        "max_objects_per_roi": max_objects,
    }
    write_manifest(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
