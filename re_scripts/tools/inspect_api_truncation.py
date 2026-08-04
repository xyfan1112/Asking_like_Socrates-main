#!/usr/bin/env python3
"""Inspect real finish_reason values before any max_tokens change."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--mode", default="debug", choices=("debug", "full"))
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    cfg = json.loads(args.settings.read_text(encoding="utf-8"))
    raw = Path(cfg["official_socratic"]["raw_output_dir"])
    candidates = list(raw.glob(f"local_api_calls_*_{args.mode}_*.jsonl"))
    if not candidates:
        print(f"[FAIL] no API log found in {raw}")
        return 2
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    finish = Counter()
    length_by_role = Counter()
    suspicious = []
    calls = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "api_call":
            continue
        calls += 1
        reason = str(row.get("finish_reason") or "unknown").casefold()
        finish[reason] += 1
        role = str(row.get("role") or row.get("model") or "unknown")
        if reason == "length":
            length_by_role[role] += 1
        text = str(row.get("response_text") or row.get("response_preview") or "")
        low = text.rstrip().casefold()
        if reason == "length" or len(text.strip()) < 30 or low.endswith(("let’s look at the im", "let's look at the im", "</que", "</ques", "</thin")):
            if len(suspicious) < args.limit:
                suspicious.append({
                    "line": line_no,
                    "role": role,
                    "call_kind": row.get("call_kind"),
                    "finish_reason": reason,
                    "prompt_tokens": row.get("prompt_tokens"),
                    "completion_tokens": row.get("completion_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "response_preview": text[:300],
                })
    result = {
        "api_log": str(path),
        "calls": calls,
        "finish_reason_counts": dict(finish),
        "length_by_role": dict(length_by_role),
        "suspicious_examples": suspicious,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if sum(length_by_role.values()) == 0:
        print("[DECISION] No finish_reason=length. Do not increase max_tokens.")
    else:
        print("[DECISION] True length truncation exists. Review only the affected role before changing its max_tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
