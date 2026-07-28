#!/usr/bin/env python3
"""Print and verify the newest v4.3.3 Socratic debug audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _zero(api: dict[str, Any], key: str) -> bool:
    return int(api.get(key, 0) or 0) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    raw_dir = Path(settings["official_socratic"]["raw_output_dir"])
    candidates = [
        path
        for path in raw_dir.glob(
            f"dota128_{args.split}_official_debug_*.audit.json"
        )
        if path.is_file()
    ]
    if not candidates:
        print(f"[FAIL] no debug audit found in {raw_dir}")
        return 2
    audit_path = max(candidates, key=lambda path: path.stat().st_mtime)
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    api = report.get("api") if isinstance(report.get("api"), dict) else {}
    rates = (
        report.get("strict_rate_by_task")
        if isinstance(report.get("strict_rate_by_task"), dict)
        else {}
    )
    thresholds = (
        report.get("thresholds")
        if isinstance(report.get("thresholds"), dict)
        else {}
    )
    strict = (
        report.get("strict_candidates_by_task")
        if isinstance(report.get("strict_candidates_by_task"), dict)
        else {}
    )
    grounding_gates = (
        report.get("grounding_gate_counts")
        if isinstance(report.get("grounding_gate_counts"), dict)
        else {}
    )

    checks = {
        "schema_is_v4_3_3": report.get("schema_version")
        == "official_debug_quality_gate_v4_3_3",
        "status_is_pass": report.get("status") == "PASS",
        "api_errors_zero": _zero(api, "api_errors"),
        "length_truncation_zero": _zero(api, "length_truncation"),
        "unrepaired_zero": _zero(api, "unrepaired"),
        "coordinate_mutations_zero": int(
            report.get("coordinate_rewrite_mutations", 0) or 0
        )
        == 0,
        "perceiver_context_missing_zero": _zero(
            api, "perceiver_context_missing"
        ),
        "perceiver_focus_roi_missing_zero": _zero(
            api, "perceiver_focus_roi_missing"
        ),
        "duplicate_questions_zero": int(
            report.get("duplicate_question_trajectories", 0) or 0
        )
        == 0,
        "classification_contradictions_zero": int(
            report.get("classification_contradiction_trajectories", 0) or 0
        )
        == 0,
        "classification_leading_questions_zero": int(
            report.get("classification_leading_question_trajectories", 0) or 0
        )
        == 0,
        "grounding_rate_met": float(rates.get("ref_grounding_obb", 0.0) or 0.0)
        >= float(thresholds.get("grounding_strict_rate_min", 0.30) or 0.30),
        "classification_rate_met": float(
            rates.get("ref_classification", 0.0) or 0.0
        )
        >= float(
            thresholds.get("classification_strict_rate_min", 0.70) or 0.70
        ),
        "strict_grounding_passes_both_geometry_gates": int(
            strict.get("ref_grounding_obb", 0) or 0
        )
        <= int(grounding_gates.get("both_pass", 0) or 0),
        "no_hard_failures": not report.get("hard_failures"),
        "no_quality_failures": not report.get("quality_failures"),
    }
    payload = {
        "audit": str(audit_path),
        "status": report.get("status"),
        "strict_rates": rates,
        "strict_candidates": strict,
        "grounding_gates": grounding_gates,
        "api": api,
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"[FAIL] debug40 is blocked: {failed}")
        return 2
    print("[PASS] debug40 meets every v4.3.3 gate; full/fresh is allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
