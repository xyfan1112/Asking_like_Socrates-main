#!/usr/bin/env python3
"""Audit raw/strict/merge Socratic artifacts without loading a model.

This tool answers three practical questions:
1. Is the canonical raw complete and internally consistent?
2. Did Grounding survive strict filtering, or is the result classification-only?
3. Do queries/traces contain known v4.2 failure patterns?
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Bad JSONL {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return [row for row in value if isinstance(row, dict)]


def raw_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter(str(row.get("task", "unknown")) for row in rows)
    success = Counter()
    errors = Counter()
    verifier_rejects = Counter()
    rewritten_changed = 0
    coordinate_mutations = 0
    query_patterns = Counter()

    for row in rows:
        task = str(row.get("task", "unknown"))
        loop = row.get("loop_result") if isinstance(row.get("loop_result"), dict) else {}
        if loop.get("success"):
            success[task] += 1
        else:
            error = str(loop.get("error") or "")
            if error:
                errors[error] += 1
            else:
                reasons = [
                    str(turn.get("V_reason"))
                    for turn in (loop.get("chat_history") or [])
                    if turn.get("V_decision") == "REJECT" and turn.get("V_reason")
                ]
                verifier_rejects[reasons[-1] if reasons else "verifier_reject_unspecified"] += 1

        query = str(row.get("query") or "")
        rewritten = str(row.get("rewritten_query") or "")
        if rewritten.strip() != query.strip():
            rewritten_changed += 1
        if "[0,1000]" in query and "[0,1000]" not in rewritten:
            coordinate_mutations += 1
        lower = query.lower()
        # Inspect the referring phrase only.  The classification query also lists
        # every DOTA label, so searching the entire string would falsely count
        # storage tank/roundabout/harbor on unrelated samples.
        if "locate " in lower and ". use normalized" in lower:
            target_phrase = lower.split("locate ", 1)[1].split(". use normalized", 1)[0]
        elif "best matches " in lower:
            target_phrase = lower.split("best matches ", 1)[1].split("?", 1)[0]
        else:
            target_phrase = lower
        patterns = {
            "of_its_class": "of its class" in target_phrase,
            "storage_tank_orientation": "storage tank" in target_phrase and any(x in target_phrase for x in ("horizontal", "vertical", "diagonally")),
            "roundabout_orientation": "roundabout" in target_phrase and any(x in target_phrase for x in ("horizontal", "vertical", "diagonally")),
            "harbor_compact_geometry": "harbor" in target_phrase and any(x in target_phrase for x in ("elongated", "horizontal", "vertical", "diagonally")),
        }
        for name, present in patterns.items():
            if present:
                query_patterns[name] += 1

    by_task: dict[str, Any] = {}
    for task, total in tasks.items():
        s = success.get(task, 0)
        by_task[task] = {
            "total": total,
            "success": s,
            "failure": total - s,
            "success_rate": round(s / total, 6) if total else 0.0,
        }
    return {
        "rows": len(rows),
        "tasks": dict(tasks),
        "by_task": by_task,
        "loop_errors": dict(errors.most_common()),
        "verifier_rejects": dict(verifier_rejects.most_common(20)),
        "rewritten_changed": rewritten_changed,
        "coordinate_mutations": coordinate_mutations,
        "query_patterns": dict(query_patterns),
    }


def trace_contradiction_count(rows: list[dict[str, Any]]) -> tuple[int, list[str]]:
    count = 0
    examples: list[str] = []
    for row in rows:
        task = str(row.get("task", ""))
        if task != "ref_classification":
            continue
        text = str(row.get("thinking") or "").lower()
        answer = str(row.get("answer") or row.get("raw_gt") or "").lower()
        bad = False
        if answer in {"large vehicle", "small vehicle"} and any(
            phrase in text
            for phrase in (
                "not a road vehicle",
                "appears to be a building",
                "industrial building",
                "part of a larger structure",
                "rather than a vehicle",
            )
        ):
            bad = True
        if answer == "ship" and any(
            phrase in text
            for phrase in ("clearly a pier", "object is a pier", "not a ship", "dock, not")
        ):
            bad = True
        if answer == "plane" and "object appears to be a runway" in text and "wings" not in text:
            bad = True
        if bad:
            count += 1
            if len(examples) < 10:
                examples.append(str(row.get("id")))
    return count, examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--strict")
    parser.add_argument("--merge")
    parser.add_argument("--output")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    strict_path = Path(args.strict) if args.strict else raw_path.with_name(raw_path.stem + "_strict.jsonl")
    merge_path = Path(args.merge) if args.merge else raw_path.parent.parent / "postproc" / (raw_path.stem + "_merge.json")

    raw_rows = read_jsonl(raw_path)
    strict_rows = read_jsonl(strict_path)
    merge_rows = read_json_list(merge_path)
    contradiction_count, contradiction_examples = trace_contradiction_count(merge_rows)

    report = {
        "raw": raw_report(raw_rows),
        "strict": {
            "path": str(strict_path),
            "rows": len(strict_rows),
            "tasks": dict(Counter(str(row.get("task", "unknown")) for row in strict_rows)),
        },
        "merge": {
            "path": str(merge_path),
            "rows": len(merge_rows),
            "tasks": dict(Counter(str(row.get("task", "unknown")) for row in merge_rows)),
            "heuristic_contradictory_classification_traces": contradiction_count,
            "contradiction_example_ids": contradiction_examples,
        },
    }
    grounding_strict = report["strict"]["tasks"].get("ref_grounding_obb", 0)
    classification_strict = report["strict"]["tasks"].get("ref_classification", 0)
    report["usable_for_b2"] = bool(grounding_strict >= 20 and classification_strict >= 20 and contradiction_count == 0)

    output = Path(args.output) if args.output else raw_path.with_suffix(".artifact_audit.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SOCRATIC ARTIFACT AUDIT] {'PASS' if report['usable_for_b2'] else 'FAIL'}")
    print(f"  raw={report['raw']['rows']} by_task={report['raw']['by_task']}")
    print(f"  coordinate_mutations={report['raw']['coordinate_mutations']} query_patterns={report['raw']['query_patterns']}")
    print(f"  strict={report['strict']['rows']} tasks={report['strict']['tasks']}")
    print(f"  merge={report['merge']['rows']} tasks={report['merge']['tasks']}")
    print(f"  contradictory_merge_traces={contradiction_count}")
    print(f"  report: {output}")
    return 0 if report["usable_for_b2"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
