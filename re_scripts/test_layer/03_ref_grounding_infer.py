#!/usr/bin/env python3
"""Grounding inference for DOTA-Ref OBB and VRSBench-Ref HBB."""
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
    append_jsonl, hbb_from_points, image_data_url, load_settings, parse_hbb_output,
    parse_obb_output_status, read_jsonl, settings_from_cli,
)
from eval_common import chat_once, check_server, protocol, write_manifest  # noqa: E402


def _first(obj: dict, *keys):
    for key in keys:
        if key in obj and obj[key] not in (None, "", []):
            return obj[key]
    return None


def load_dota(settings: dict, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(settings["paths"]["dota128_ref_root"]) / f"{split}.jsonl")
    return [{
        "id": r["id"], "dataset": "dota_ref", "image_path": r["image_path"],
        "scene_id": r.get("scene_id") or Path(r["image_path"]).stem.split("__", 1)[0],
        "width": r["image_width"], "height": r["image_height"],
        "question": r.get("grounding_question") or r["question"], "reference": r.get("reference_grounding") or r.get("reference"),
        "gt_class": r["class_name"], "gt_obb": r.get("obb_pixel") or r.get("obj_corner_pixel"),
        "gt_hbb": r.get("hbb_pixel") or hbb_from_points(r.get("obb_pixel") or r.get("obj_corner_pixel")),
        "target_type": "obb", "coordinate_mode": r.get("coordinate_target", "norm1000_obb"),
        "ref_group": (
            "unique"
            if int((r.get("reference_quality") or {}).get("same_class_count", 1)) == 1
            else "nonunique"
        ),
    } for r in rows]


def load_vrsbench(settings: dict) -> list[dict[str, Any]]:
    path = Path(settings["paths"]["vrsbench_ref_json"])
    if not path.is_file():
        raise FileNotFoundError(f"VRSBench-Ref file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "annotations", "samples", "items"):
            if isinstance(value.get(key), list):
                value = value[key]; break
    if not isinstance(value, list):
        raise ValueError("Unsupported VRSBench-Ref JSON: expected a list or a dict containing data/annotations/samples")
    image_root = Path(settings["paths"]["vrsbench_image_dir"])
    rows = []
    for index, item in enumerate(value):
        image_name = _first(item, "image", "image_name", "image_path", "filename", "file_name")
        reference = _first(item, "reference", "expression", "referring_expression", "question", "text")
        bbox = _first(item, "bbox", "box", "gt_bbox")
        if isinstance(bbox, dict):
            bbox = _first(bbox, "bbox", "box", "coordinates")
        if not image_name or not reference or not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        image_path = Path(str(image_name))
        if not image_path.is_absolute():
            image_path = image_root / image_path.name
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        # VRSBench public grounding boxes are commonly normalized. Detect based on range.
        maxv = max(abs(float(x)) for x in bbox)
        if maxv <= 1.5:
            gt_hbb = [bbox[0]*width,bbox[1]*height,bbox[2]*width,bbox[3]*height]
        elif maxv <= 100.5:
            gt_hbb = [bbox[0]/100*width,bbox[1]/100*height,bbox[2]/100*width,bbox[3]/100*height]
        elif maxv <= 1000.5:
            gt_hbb = [bbox[0]/1000*width,bbox[1]/1000*height,bbox[2]/1000*width,bbox[3]/1000*height]
        else:
            gt_hbb = [float(x) for x in bbox]
        question = (
            f"Locate {reference}. Return the axis-aligned bounding box normalized to [0,1000] as "
            'Answer: {"bbox": [x1, y1, x2, y2]}.'
        )
        rows.append({
            "id": str(_first(item, "id", "ref_id", "question_id") or f"vrsref-{index}"),
            "dataset": "vrsbench_ref", "image_path": str(image_path), "width": width, "height": height,
            "question": question, "reference": str(reference), "gt_class": _first(item, "class_name", "category", "label"),
            "gt_obb": None, "gt_hbb": gt_hbb, "target_type": "hbb", "coordinate_mode": "norm1000_hbb",
        })
    if not rows:
        raise RuntimeError("No usable VRSBench-Ref rows were parsed. Inspect the local JSON schema and adapt load_vrsbench().")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--model-key", default="rs_eot")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--dataset", choices=["dota_ref", "vrsbench_ref"], default="dota_ref")
    ap.add_argument("--split", default="val")
    ap.add_argument("--protocol", choices=["ref_grounding", "ref_grounding_k5"], default="ref_grounding")
    ap.add_argument("--k", type=int)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--output")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    model = settings["models"][args.model_key]
    cfg = protocol(settings, args.protocol)
    k = args.k or int(cfg.get("k", 1))
    check_server(args.base_url, model["served_name"])
    samples = load_dota(settings, args.split) if args.dataset == "dota_ref" else load_vrsbench(settings)
    if args.max_samples:
        samples = samples[: args.max_samples]
    output = Path(args.output) if args.output else Path(settings["paths"]["test_run_root"]) / "ref_grounding" / args.dataset / args.model_key / f"{args.split}_{args.protocol}_k{k}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        output.unlink(missing_ok=True)
        output.with_suffix(".manifest.json").unlink(missing_ok=True)
    done = set()
    if output.exists():
        done = {(r.get("id"), int(r.get("run_id", 0))) for r in read_jsonl(output, skip_bad=True)}

    for index, sample in enumerate(samples, 1):
        for run_id in range(k):
            if (sample["id"], run_id) in done:
                continue
            response = chat_once(
                args.base_url, model["served_name"],
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(sample["image_path"])}},
                    {"type": "text", "text": sample["question"]},
                ]}],
                temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]), max_tokens=int(cfg["max_tokens"]), retries=1,
            )
            if sample["target_type"] == "obb":
                parsed = parse_obb_output_status(
                    response["raw_output"],
                    sample["width"],
                    sample["height"],
                    cfg.get("coordinate_mode", "normalized_0_1000"),
                    strict=True,
                    exactly_one=True,
                )
                one = parsed["predictions"][0] if parsed["predictions"] else None
                pred_class = one["class_name"] if one else None
                pred_obb = one["points"] if one else None
                pred_hbb = hbb_from_points(pred_obb) if pred_obb else None
                detected = one.get("coordinate_mode_detected") if one else None
                parse_meta = {
                    "format_ok": parsed["format_ok"],
                    "parse_ok": parsed["parse_ok"],
                    "parse_reason": parsed["reason"],
                }
            else:
                one = parse_hbb_output(response["raw_output"], sample["width"], sample["height"], cfg.get("coordinate_mode", "normalized_0_1000"))
                pred_class = one.get("class_name") if one else None
                pred_obb = None; pred_hbb = one["bbox"] if one else None; detected = one.get("coordinate_mode_detected") if one else None
                parse_meta = {
                    "format_ok": one is not None,
                    "parse_ok": one is not None,
                    "parse_reason": None if one is not None else "unparseable_hbb",
                }
            append_jsonl(output, {
                **sample, "run_id": run_id, "model_key": args.model_key, "protocol": args.protocol,
                "pred_class": pred_class, "pred_obb": pred_obb, "pred_hbb": pred_hbb,
                "coordinate_mode_detected": detected, **parse_meta, **response,
            })
            print(f"[{index}/{len(samples)}] run={run_id} parsed={pred_hbb is not None} error={bool(response['error'])}")

    manifest = {"dataset": args.dataset, "model_key": args.model_key, "split": args.split, "protocol": args.protocol, "k": k,
                "questions": len(samples), "expected_runs": len(samples)*k, "output": str(output), "settings_protocol": cfg}
    write_manifest(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
