#!/usr/bin/env python3
"""Create balanced visual previews for manual DOTA-Ref inspection."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from collections import defaultdict, deque
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import load_settings, read_jsonl, settings_from_cli, write_json  # noqa: E402


def _balanced(rows: list[dict], limit: int) -> list[dict]:
    """Round-robin by class, then by image, for more useful previews."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("class_name", "unknown"))].append(row)
    queues = {k: deque(sorted(v, key=lambda r: (r.get("image_name", ""), r.get("object_index", 0)))) for k, v in grouped.items()}
    keys = sorted(queues)
    output: list[dict] = []
    while keys and len(output) < limit:
        next_keys = []
        for key in keys:
            if queues[key] and len(output) < limit:
                output.append(queues[key].popleft())
            if queues[key]:
                next_keys.append(key)
        keys = next_keys
    return output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--train-limit", type=int, default=30)
    ap.add_argument("--val-limit", type=int, default=20)
    ap.add_argument("--keep-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    ref_root = Path(settings["paths"]["dota128_ref_root"])
    out = Path(settings["paths"]["pipeline_work_root"]) / "ref_previews"
    if out.exists() and not args.keep_existing:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "ref_preview_v4_2",
        "output_dir": str(out),
        "splits": {},
        "failed": [],
    }
    generated = 0

    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        source = ref_root / f"{split}.jsonl"
        if not source.is_file():
            report["failed"].append(f"missing source: {source}")
            report["splits"][split] = {"requested": limit, "generated": 0, "source": str(source)}
            continue
        rows = _balanced(read_jsonl(source), max(0, limit))
        split_generated = 0
        examples = []
        for row in rows:
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                draw = ImageDraw.Draw(image)
                points = [tuple(int(round(v)) for v in point) for point in row["obb_pixel"]]
                draw.line(points + [points[0]], fill=(255, 0, 0), width=max(2, min(image.size) // 250))
                label = (
                    f"class: {row['class_name']}\n"
                    f"grounding ref: {row['reference_grounding']}\n"
                    f"classification ref: {row['reference_classification']}"
                )
                wrapped = "\n".join(
                    line for paragraph in label.splitlines() for line in textwrap.wrap(paragraph, 95)
                )
                text_height = 120
                canvas = Image.new("RGB", (image.width, image.height + text_height), "white")
                canvas.paste(image, (0, text_height))
                ImageDraw.Draw(canvas).multiline_text((6, 5), wrapped, fill="black", spacing=3)
                target = out / f"{split}_{row['id'][:12]}.png"
                canvas.save(target)
                generated += 1
                split_generated += 1
                if len(examples) < 5:
                    examples.append(str(target))
            except Exception as exc:
                report["failed"].append(
                    f"{split}:{row.get('id')}: {type(exc).__name__}: {exc}"
                )
        report["splits"][split] = {
            "source": str(source),
            "requested": limit,
            "generated": split_generated,
            "examples": examples,
        }

    report["generated_total"] = generated
    report["passed"] = generated > 0 and not report["failed"]
    report["manual_check"] = [
        "red OBB covers exactly the referred object",
        "grounding reference uniquely identifies the red object",
        "classification reference does not reveal the canonical DOTA label",
        "no preview contains an out-of-image box",
    ]
    report["next_step"] = (
        "manually inspect previews; then run python data_layer/05_build_agent_inputs.py"
        if report["passed"]
        else "fix failed preview rows before building agent inputs"
    )
    report_path = out / "preview_report.json"
    write_json(report_path, report)

    print(f"[REF PREVIEW] {'PASS' if report['passed'] else 'FAIL'}")
    for split, item in report["splits"].items():
        print(f"  {split}: generated={item['generated']}/{item['requested']}")
    print(f"  output: {out}")
    print(f"  report: {report_path}")
    print("  manual check: red box / unique grounding ref / masked classification ref")
    print(f"  next: {report['next_step']}")
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
