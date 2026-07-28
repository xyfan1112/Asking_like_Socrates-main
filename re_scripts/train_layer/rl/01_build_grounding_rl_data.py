#!/usr/bin/env python3
"""Build EasyR1-compatible grounding data from selected DOTA-Ref samples.

The official paper's first RL stage is grounding. The bundled EasyR1 examples
use normalized HBB rewards, so ``norm1000_hbb`` is the primary compatible label.
The original OBB remains in ground truth for later experimental rewards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import load_settings, read_jsonl, settings_from_cli, write_json  # noqa: E402


def build_row(row: dict) -> dict:
    return {
        "problem": (
            "Locate the object described by this referring expression: "
            f"{row['reference_grounding']}"
        ),
        "images": [row["image_path"]],
        "answer": {
            "class_name": row["class_name"],
            "bbox": [int(round(float(x))) for x in row["hbb_norm1000"]],
            "obb": [int(round(float(v))) for p in row["obb_norm1000"] for v in p],
            "coordinate_range": [0, 1000],
            "source_id": row["id"],
        },
        "metadata": {
            "task": "dota_ref_grounding_rl",
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "reference": row["reference_grounding"],
            "class_id": row["class_id"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    ref_root = Path(settings["paths"]["dota128_ref_root"])
    out_root = Path(settings["paths"]["dota128_rl_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"format": "EasyR1 JSONL", "primary_label": "norm1000_hbb", "splits": {}}
    for split in settings["data_conversion"].get("splits", ["train", "val"]):
        rows = read_jsonl(ref_root / f"{split}.jsonl")
        output = out_root / f"{split}.jsonl"
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(build_row(row), ensure_ascii=False) + "\n")
        report["splits"][split] = {"rows": len(rows), "file": str(output)}
    write_json(out_root / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
