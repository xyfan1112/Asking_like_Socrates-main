#!/usr/bin/env python3
"""Create one compact inventory/funnel report from current v1.2.3 artifacts.

This does not replace strict gates. It only gathers their existing counters and
file counts into a convenient analysis index and never fabricates missing data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def json_count(path: Path) -> int | None:
    value = load_json(path)
    if isinstance(value, list):
        return len(value)
    return None


def jsonl_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def newest(root: Path, patterns: list[str]) -> Path | None:
    rows: list[Path] = []
    for pattern in patterns:
        rows.extend(root.glob(pattern))
    rows = [p for p in rows if p.is_file()]
    return max(rows, key=lambda p: p.stat().st_mtime) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    settings_path = args.settings.expanduser().resolve()
    settings = load_json(settings_path)
    if not isinstance(settings, dict):
        raise SystemExit(f"Invalid settings: {settings_path}")

    paths = settings["paths"]
    pipeline = Path(paths["pipeline_work_root"])
    lf = Path(paths["dota128_llamafactory_root"])
    raw = Path(settings.get("official_socratic", {}).get("raw_output_dir", pipeline / "official_socratic" / "raw"))
    post = Path(settings.get("official_socratic", {}).get("postproc_output_dir", pipeline / "official_socratic" / "postproc"))
    reports = pipeline / "reports"

    files = {
        "train_agent_inputs": pipeline / "agent_inputs" / "train_agent_inputs.jsonl",
        "val_agent_inputs": pipeline / "agent_inputs" / "val_agent_inputs.jsonl",
        "train_direct": pipeline / "agent_inputs" / "train_direct.json",
        "val_direct": pipeline / "agent_inputs" / "val_direct.json",
        "lf_direct_train": lf / "dota128_direct_train_official.json",
        "lf_b1_matched_train": lf / "dota128_b1_matched_train_official.json",
        "lf_b2_matched_train": lf / "dota128_b2_matched_train_official.json",
        "lf_socratic_train": lf / "dota128_socratic_train_official.json",
    }
    artifact_counts: dict[str, Any] = {}
    for key, path in files.items():
        count = jsonl_count(path) if path.suffix == ".jsonl" else json_count(path)
        artifact_counts[key] = {"path": str(path), "exists": path.is_file(), "rows": count}

    debug_audit = newest(raw, ["*.audit.json"])
    full_audit = newest(raw, ["*.full_audit.json"])
    strict_report = newest(raw, ["*strict.report.json"])
    conversion_report = newest(post, ["*conversion_report*.json", "*convert*.report.json", "*.report.json"])

    source_reports: dict[str, Any] = {}
    for name, path in {
        "debug_audit": debug_audit,
        "full_audit": full_audit,
        "strict_report": strict_report,
        "conversion_report": conversion_report,
        "lineage_audit": reports / "lineage_audit.json",
        "data_layer_final_report": reports / "data_layer_final_report.json",
    }.items():
        if path and path.is_file():
            source_reports[name] = {"path": str(path), "content": load_json(path)}
        else:
            source_reports[name] = {"path": str(path) if path else None, "content": None}

    rejection_counts: Counter[str] = Counter()
    for key in ("debug_audit", "strict_report"):
        content = source_reports[key]["content"]
        if isinstance(content, dict):
            values = content.get("strict_rejection_reasons") or content.get("rejection_reasons") or {}
            if isinstance(values, dict):
                for reason, count in values.items():
                    try:
                        rejection_counts[str(reason)] += int(count)
                    except Exception:
                        pass

    strict = source_reports["strict_report"]["content"]
    funnel = {
        "attempted": None,
        "official_success": None,
        "strict_input": None,
        "strict_accepted": None,
        "strict_rejected": None,
        "b2_socratic_train_rows": artifact_counts["lf_socratic_train"]["rows"],
    }
    dbg = source_reports["debug_audit"]["content"]
    if isinstance(dbg, dict):
        funnel["attempted"] = dbg.get("attempted")
        funnel["official_success"] = dbg.get("official_success")
    if isinstance(strict, dict):
        funnel["strict_input"] = strict.get("input_rows")
        funnel["strict_accepted"] = strict.get("accepted_strict")
        funnel["strict_rejected"] = strict.get("rejected")

    report = {
        "schema_version": "pipeline_summary_v1_2_3",
        "settings": str(settings_path),
        "version": settings.get("v1_2_3", {}).get("version"),
        "language": settings.get("v1_2_3", {}).get("language"),
        "dataset_root": settings.get("v1_2_3", {}).get("dataset_root"),
        "artifact_counts": artifact_counts,
        "socratic_funnel": funnel,
        "rejection_reasons_aggregated": dict(rejection_counts.most_common()),
        "source_reports": source_reports,
        "notes": [
            "Missing values mean the corresponding stage has not produced an artifact; they are not treated as zero.",
            "Strict gate reports remain authoritative; this file is an index for analysis and support bundles.",
        ],
    }
    output = args.output or (reports / "pipeline_summary_v1_2_3.json")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[PIPELINE SUMMARY] PASS")
    print("  output =", output)
    print("  counts =", {k: v["rows"] for k, v in artifact_counts.items()})
    print("  funnel =", funnel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
