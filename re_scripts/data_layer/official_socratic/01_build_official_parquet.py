#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main_layer"))
from common import load_settings, read_jsonl, settings_from_cli  # noqa: E402


def ensure_image_marker(text: str) -> str:
    text = str(text or "").strip()
    return text if text.lower().startswith("<image>") else f"<image>\n{text}"


def build_record(item: Dict[str, Any]) -> Dict[str, Any]:
    image_path = Path(item["image_path"]).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    return {
        "id": str(item["id"]),
        "query": ensure_image_marker(item["query"]),
        "gt": str(item.get("gt", "")).strip(),
        "image": [["rgb", image_path.name]],
        "image_root": {"rgb": str(image_path.parent)},
        "data_source": str(item.get("data_source", "DOTA128-Ref")),
        "task": str(item.get("task", "ref_grounding_obb")),
        "lang": str(item.get("qa_language") or item.get("lang") or "en"),
        "qa_language": str(item.get("qa_language") or item.get("lang") or "en"),
        "taxonomy_sha256": str(item.get("taxonomy_sha256") or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert re_scripts agent inputs to the exact Parquet schema expected by official SocraticAgent/generation.py."
    )
    parser.add_argument("--settings", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true", help="忽略数据层最终门控，仅用于调试")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    work_root = Path(settings["paths"]["pipeline_work_root"])
    input_path = Path(args.input) if args.input else work_root / "agent_inputs" / f"{args.split}_agent_inputs.jsonl"
    output_dir = Path(settings["official_socratic"]["input_parquet_dir"])
    output_path = Path(args.output) if args.output else output_dir / f"dota128_{args.split}_official.parquet"

    final_gate = work_root / "reports" / "data_layer_final_report.json"
    if not final_gate.is_file() and not args.force:
        raise FileNotFoundError(
            f"缺少数据层最终门控报告: {final_gate}\n"
            "先运行: python main_layer/run.py data-full --settings settings.json"
        )
    if final_gate.is_file() and not args.force:
        gate = json.loads(final_gate.read_text(encoding="utf-8"))
        if not gate.get("passed", False):
            raise RuntimeError(
                f"数据层最终门控未通过: {final_gate}。禁止构建官方 Parquet。"
            )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing agent input: {input_path}\n"
            "Run: python main_layer/run.py data-full --settings settings.json\n"
            "or use main_layer/run.py build-official-parquet, which auto-prepares prerequisites."
        )
    rows = read_jsonl(input_path, skip_bad=False)
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for index, item in enumerate(rows):
        try:
            records.append(build_record(item))
        except Exception as exc:  # keep a complete audit instead of silently dropping rows
            errors.append({"index": index, "id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"})

    if errors:
        audit_path = output_dir / f"dota128_{args.split}_official_parquet_errors.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"Failed to convert {len(errors)} rows. See {audit_path}")
    if not records:
        raise RuntimeError(f"No valid records found in {input_path}")

    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required in als_sft. Install it before running this script.") from exc

    dataset = datasets.Dataset.from_list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))

    report = {
        "split": args.split,
        "input": str(input_path),
        "output": str(output_path),
        "records": len(records),
        "schema_fields": list(records[0].keys()),
        "official_expected_fields": ["id", "query", "gt", "image", "image_root", "data_source", "task", "lang", "qa_language", "taxonomy_sha256"],
        "qa_languages": sorted({str(row.get("qa_language") or row.get("lang") or "") for row in rows}),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OFFICIAL PARQUET] PASS")
    print(f"  split: {args.split}")
    print(f"  records: {len(records)}")
    print(f"  input: {input_path}")
    print(f"  output: {output_path}")
    print(f"  report: {report_path}")
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
