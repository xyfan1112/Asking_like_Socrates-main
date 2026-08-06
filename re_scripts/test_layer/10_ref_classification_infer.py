#!/usr/bin/env python3
"""Exact-label classification inference on DOTA128-Ref."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
sys.path.insert(0, str(ROOT / "test_layer"))
from common import (  # noqa: E402
    append_jsonl,
    extract_final_text,
    image_data_url,
    load_settings,
    normalize_class_name,
    read_jsonl,
    settings_from_cli,
)
from eval_common import chat_once, check_server, protocol, write_manifest  # noqa: E402


def stable_shard(value: str, num_shards: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % num_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--model-key", default="rs_eot")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--output")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard arguments")
    settings = load_settings(settings_from_cli(__file__, args.settings))
    model = settings["models"][args.model_key]
    cfg = protocol(settings, "ref_classification")
    k = args.k or int(cfg.get("k", 1))
    check_server(args.base_url, model["served_name"])
    rows = read_jsonl(
        Path(settings["paths"]["dota128_ref_root"]) / f"{args.split}.jsonl"
    )
    if args.max_samples:
        rows = rows[: args.max_samples]
    rows = [row for row in rows if stable_shard(str(row["id"]), args.num_shards) == args.shard_index]
    output = (
        Path(args.output)
        if args.output
        else Path(settings["paths"]["test_run_root"])
        / "ref_classification"
        / args.model_key
        / f"{args.split}_k{k}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        output.unlink(missing_ok=True)
        output.with_suffix(".manifest.json").unlink(missing_ok=True)
    done = set()
    if output.exists():
        done = {
            (str(row.get("id")), int(row.get("run_id", 0)))
            for row in read_jsonl(output, skip_bad=True)
        }

    for index, row in enumerate(rows, 1):
        for run_id in range(k):
            if (str(row["id"]), run_id) in done:
                continue
            response = chat_once(
                args.base_url,
                model["served_name"],
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url(row["image_path"])},
                            },
                            {
                                "type": "text",
                                "text": row["classification_question"],
                            },
                        ],
                    }
                ],
                temperature=float(cfg["temperature"]),
                top_p=float(cfg["top_p"]),
                max_tokens=int(cfg["max_tokens"]),
                retries=1,
            )
            final = extract_final_text(response["raw_output"])
            pred_class = normalize_class_name(final)
            append_jsonl(
                output,
                {
                    "id": row["id"],
                    "run_id": run_id,
                    "model_key": args.model_key,
                    "image_path": row["image_path"],
                    "scene_id": row.get("scene_id")
                    or Path(row["image_path"]).stem.split("__", 1)[0],
                    "question": row["classification_question"],
                    "gt_class": row["class_name"],
                    "pred_class": pred_class,
                    "final_text": final,
                    "parse_ok": pred_class is not None,
                    "ref_group": (
                        "unique"
                        if int(
                            (row.get("reference_quality") or {}).get(
                                "same_class_count", 1
                            )
                        )
                        == 1
                        else "nonunique"
                    ),
                    **response,
                },
            )
            print(
                f"[{index}/{len(rows)}] run={run_id} gt={row['class_name']} "
                f"pred={pred_class} error={bool(response['error'])}"
            )

    manifest = {
        "schema_version": "dota_ref_classification_v1",
        "model_key": args.model_key,
        "split": args.split,
        "k": k,
        "questions": len(rows),
        "expected_runs": len(rows) * k,
        "output": str(output),
        "settings_protocol": cfg,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    write_manifest(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
