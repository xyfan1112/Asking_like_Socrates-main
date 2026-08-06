#!/usr/bin/env python3
"""Deterministically merge evaluation JSONL shards and validate completeness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"invalid JSONL {path}:{lineno}: {exc}")
            if not isinstance(value, dict):
                raise SystemExit(f"non-object JSONL row {path}:{lineno}")
            rows.append(value)
    return rows


def key_for(row: dict[str, Any]) -> tuple[str, int]:
    ident = row.get("query_id", row.get("id"))
    if ident is None:
        raise SystemExit("row lacks query_id/id")
    return str(ident), int(row.get("run_id", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--shards", required=True, nargs="+", type=Path)
    ap.add_argument("--manifest-output", type=Path)
    ap.add_argument("--delete-shards", action="store_true")
    args = ap.parse_args()

    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    shard_reports = []
    expected = 0
    for shard in args.shards:
        if not shard.is_file():
            raise SystemExit(f"missing shard: {shard}")
        manifest_path = shard.with_suffix(".manifest.json")
        manifest = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected += int(manifest.get("expected_rows", manifest.get("expected_runs", 0)) or 0)
        shard_rows = read_jsonl(shard)
        duplicate_keys = []
        for row in shard_rows:
            key = key_for(row)
            if key in rows_by_key:
                duplicate_keys.append(key)
            else:
                rows_by_key[key] = row
        if duplicate_keys:
            raise SystemExit(f"duplicate keys across shards, first examples: {duplicate_keys[:10]}")
        shard_reports.append(
            {
                "path": str(shard),
                "rows": len(shard_rows),
                "manifest": str(manifest_path) if manifest_path.is_file() else None,
                "expected": int(manifest.get("expected_rows", manifest.get("expected_runs", len(shard_rows))) or 0),
            }
        )

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    if expected and len(rows) != expected:
        raise SystemExit(f"merged row count mismatch: rows={len(rows)} expected={expected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "evaluation_replica4_merge_v1",
        "output": str(args.output),
        "rows": len(rows),
        "expected_rows": expected or len(rows),
        "passed": (not expected) or len(rows) == expected,
        "shards": shard_reports,
        "sort_key": ["query_id_or_id", "run_id"],
    }
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")
    manifest_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.delete_shards:
        for shard in args.shards:
            shard.unlink(missing_ok=True)
            shard.with_suffix(".manifest.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
