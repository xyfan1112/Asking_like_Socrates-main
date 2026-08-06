#!/usr/bin/env python3
"""Extract trainable-parameter and completion evidence from a LLaMA-Factory log."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = {
    "trainable_params": re.compile(r"trainable params\s*[:=]\s*([0-9,]+)", re.I),
    "all_params": re.compile(r"all params\s*[:=]\s*([0-9,]+)", re.I),
    "trainable_percent": re.compile(r"trainable%\s*[:=]\s*([0-9.]+)", re.I),
}


def last_match(pattern: re.Pattern[str], text: str):
    values = pattern.findall(text)
    return values[-1] if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    text = args.log.read_text(encoding="utf-8", errors="replace") if args.log.is_file() else ""
    extracted = {name: last_match(pattern, text) for name, pattern in PATTERNS.items()}
    completed_markers = [
        "train_runtime",
        "Training completed",
        "***** train metrics *****",
    ]
    completed = any(marker in text for marker in completed_markers)
    oom = "CUDA out of memory" in text or "OutOfMemoryError" in text
    nccl_error = "NCCL" in text and ("error" in text.lower() or "failed" in text.lower())
    report = {
        "schema_version": "training_log_report_v1_2_3",
        "target": args.target,
        "log": str(args.log),
        "log_exists": args.log.is_file(),
        "completed_marker_present": completed,
        "oom_detected": oom,
        "nccl_error_detected": nccl_error,
        "runtime_parameter_summary": extracted,
        "note": (
            "Counts are parsed from LLaMA-Factory after model construction. "
            "If null, inspect the log because the installed version may use different wording."
        ),
    }
    out = Path(settings["paths"]["training_run_root"]) / "reports" / f"trainable_params_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
