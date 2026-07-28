#!/usr/bin/env python3
"""Evaluate a ShareGPT/LLaMA-Factory multimodal dataset without assuming task type.

Localization rows are emitted with parsed OBB/HBB fields for grounding metrics;
ordinary QA rows are emitted with text predictions for VQA metrics. This supports
DOTA-LLaMA validation data and RS-EoT-style QA files through one loader.
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
    append_jsonl, extract_final_text, image_data_url, load_settings, parse_hbb_output,
    parse_obb_output, read_jsonl, settings_from_cli,
)
from eval_common import chat_once, check_server, protocol, write_manifest  # noqa: E402


def _role(message):
    return message.get("role") or message.get("from")


def _content(message):
    return message.get("content") if "content" in message else message.get("value")


def load_samples(path: Path, project_root: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ShareGPT dataset must be a JSON list")
    output = []
    for index, row in enumerate(data):
        messages = row.get("messages") or row.get("conversations") or []
        images = row.get("images") or row.get("image") or []
        if isinstance(images, str):
            images = [images]
        user = next((_content(m) for m in messages if _role(m) in {"user", "human"}), None)
        assistant = next((_content(m) for m in reversed(messages) if _role(m) in {"assistant", "gpt"}), None)
        if not user or not assistant or not images:
            continue
        image_path = Path(str(images[0])).expanduser()
        if not image_path.is_absolute():
            image_path = project_root / image_path
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        gt = extract_final_text(str(assistant))
        metadata = row.get("metadata") or {}
        task = str(metadata.get("task") or "")
        gt_obb = parse_obb_output(gt, width, height, "auto")
        gt_hbb = parse_hbb_output(gt, width, height, "auto")
        if gt_obb or "obb" in task or "detection" in task:
            target_type = "obb"
        elif gt_hbb or "ground" in task:
            target_type = "hbb"
        else:
            target_type = "vqa"
        output.append({
            "id": str(metadata.get("id") or row.get("id") or index),
            "question": str(user), "gt": gt, "image_path": str(image_path.resolve()),
            "width": width, "height": height, "target_type": target_type, "metadata": metadata,
            "gt_obb_objects": gt_obb, "gt_hbb": gt_hbb["bbox"] if gt_hbb else None,
        })
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--model-key", default="rs_eot")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--protocol", default="ref_grounding", choices=["ref_grounding", "ref_grounding_k5", "vrsbench_vqa_k5"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    model = settings["models"][args.model_key]
    cfg = protocol(settings, args.protocol)
    k = args.k or int(cfg.get("k", 1))
    check_server(args.base_url, model["served_name"])
    samples = load_samples(Path(args.dataset_path), Path(settings["project"]["project_root"]))
    if args.max_samples:
        samples = samples[:args.max_samples]
    output = Path(args.output) if args.output else Path(settings["paths"]["test_run_root"]) / "sharegpt_dataset" / args.model_key / f"{Path(args.dataset_path).stem}_{args.protocol}_k{k}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
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
                ]}], temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]), max_tokens=int(cfg["max_tokens"]), retries=1,
            )
            row = {**sample, "run_id": run_id, "model_key": args.model_key, **response}
            row["prediction"] = extract_final_text(response["raw_output"])
            if sample["target_type"] == "obb":
                row["pred_obb_objects"] = parse_obb_output(response["raw_output"], sample["width"], sample["height"], cfg.get("coordinate_mode", "auto"))
            elif sample["target_type"] == "hbb":
                parsed = parse_hbb_output(response["raw_output"], sample["width"], sample["height"], cfg.get("coordinate_mode", "auto"))
                row["pred_hbb"] = parsed["bbox"] if parsed else None
            append_jsonl(output, row)
            print(f"[{index}/{len(samples)}] task={sample['target_type']} run={run_id} error={bool(response['error'])}")
    counts = {name: sum(x["target_type"] == name for x in samples) for name in ("obb", "hbb", "vqa")}
    manifest = {"dataset_path": str(Path(args.dataset_path).resolve()), "model_key": args.model_key, "protocol": args.protocol,
                "k": k, "samples": len(samples), "expected_runs": len(samples)*k, "task_counts": counts, "output": str(output)}
    write_manifest(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
