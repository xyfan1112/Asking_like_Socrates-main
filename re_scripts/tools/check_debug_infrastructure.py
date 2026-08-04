#!/usr/bin/env python3
"""Block Full until the newest Debug has healthy language and ROI infrastructure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("no matching file")
    return max(paths, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    cfg = json.loads(args.settings.read_text(encoding="utf-8"))
    expected_lang = str(cfg.get("taxonomy", {}).get("qa_language") or cfg.get("trajectory", {}).get("qa_language") or "")
    raw_dir = Path(cfg["official_socratic"]["raw_output_dir"])
    audit = _latest(list(raw_dir.glob(f"dota128_{args.split}_official_debug_*.audit.json")))
    report = json.loads(audit.read_text(encoding="utf-8"))
    api: dict[str, Any] = report.get("api") if isinstance(report.get("api"), dict) else {}

    zero_keys = (
        "api_errors",
        "perceiver_context_missing",
        "perceiver_focus_roi_missing",
        "classification_focus_roi_missing",
        "focus_crop_missing",
    )
    checks: dict[str, bool] = {
        "expected_language_is_zh": expected_lang == "zh",
        **{f"{key}_zero": int(api.get(key, 0) or 0) == 0 for key in zero_keys},
        "focus_crop_calls_positive": int(api.get("focus_crop_calls", 0) or 0) > 0,
        "focus_crop_calls_cover_perceiver_calls": int(api.get("focus_crop_calls", 0) or 0)
        >= int(api.get("perceiver_calls", 0) or 0),
    }
    payload = {
        "audit": str(audit),
        "settings_language": expected_lang,
        "status": report.get("status"),
        "attempted": report.get("attempted"),
        "official_success": report.get("official_success"),
        "api": api,
        "checks": checks,
        "token_policy": (
            "Keep current max_tokens" if int(api.get("length_truncation", 0) or 0) == 0
            else "Inspect API finish_reason=length before changing only the affected role"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("[BLOCKED] Debug infrastructure/quality is not healthy:", failed)
        print("[RULE] Do not run Full and do not lower quality thresholds.")
        return 2
    print("[PASS] Debug infrastructure is healthy. Run the separate formal Debug quality gate next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
