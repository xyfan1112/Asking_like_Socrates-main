#!/usr/bin/env python3
"""Fail-closed audit that B1 Direct supervision is unchanged inside B2.

B1-Direct-Standalone trains on ``dota128_direct_train_official.json``.
B2 trains on ``dota128_b2_matched_train_official.json`` which, by contract,
must begin with an exact copy of every Direct row before the accepted Socratic
rows are appended. This audit makes that scientific control explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha(row: Any) -> str:
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_list(path: Path) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"Expected a JSON list: {path}")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    settings_path = args.settings.expanduser().resolve()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    root = Path(settings["paths"]["dota128_llamafactory_root"]).expanduser().resolve()
    direct_path = root / "dota128_direct_train_official.json"
    b2_path = root / "dota128_b2_matched_train_official.json"
    direct = load_list(direct_path)
    b2 = load_list(b2_path)

    prefix = b2[: len(direct)]
    mismatch_indices: list[int] = []
    for idx, (left, right) in enumerate(zip(direct, prefix)):
        if left != right:
            mismatch_indices.append(idx)
            if len(mismatch_indices) >= 50:
                break

    direct_hashes = [canonical_sha(row) for row in direct]
    prefix_hashes = [canonical_sha(row) for row in prefix]
    exact = len(b2) >= len(direct) and direct == prefix
    socratic_rows = max(0, len(b2) - len(direct))
    report = {
        "schema_version": "b1_b2_direct_pairing_v1_2_3",
        "settings": str(settings_path),
        "direct_file": str(direct_path),
        "b2_file": str(b2_path),
        "direct_rows": len(direct),
        "b2_total_rows": len(b2),
        "b2_socratic_rows": socratic_rows,
        "b2_has_full_direct_prefix": len(b2) >= len(direct),
        "direct_prefix_exact": exact,
        "direct_order_preserved": direct_hashes == prefix_hashes,
        "first_mismatch_indices": mismatch_indices,
        "direct_sequence_sha256": hashlib.sha256("\n".join(direct_hashes).encode()).hexdigest(),
        "b2_prefix_sequence_sha256": hashlib.sha256("\n".join(prefix_hashes).encode()).hexdigest(),
        "passed": bool(exact and socratic_rows > 0),
        "failure_reasons": [],
    }
    if len(b2) < len(direct):
        report["failure_reasons"].append("b2_has_fewer_rows_than_direct")
    if not exact:
        report["failure_reasons"].append("b2_direct_prefix_differs_from_b1_direct")
    if socratic_rows <= 0:
        report["failure_reasons"].append("b2_contains_no_appended_socratic_rows")

    output = args.output
    if output is None:
        output = Path(settings["paths"]["training_run_root"]) / "reports" / "b1_b2_direct_pairing_report.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[B1/B2 DIRECT PAIRING] {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  direct_rows      = {len(direct)}")
    print(f"  b2_total_rows    = {len(b2)}")
    print(f"  b2_socratic_rows = {socratic_rows}")
    print(f"  exact_prefix     = {exact}")
    print(f"  report           = {output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
