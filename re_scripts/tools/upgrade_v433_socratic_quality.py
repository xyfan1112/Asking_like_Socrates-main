#!/usr/bin/env python3
"""Safely upgrade a user-specific settings file to the v4.3.3 quality gates."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


def _replace_version(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_version(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_version(item) for item in value]
    if isinstance(value, str):
        return value.replace("4.3.2", "4.3.3")
    return value


def upgrade(settings: dict[str, Any], bump_output_version: bool) -> dict[str, Any]:
    result = _replace_version(deepcopy(settings)) if bump_output_version else deepcopy(settings)
    conversion = result.setdefault("data_conversion", {})
    conversion["socratic_min_short_side_norm1000"] = 20.0
    conversion["socratic_min_area_ratio"] = 0.0002
    conversion["require_unique_reference"] = True

    trajectory = result.setdefault("trajectory", {})
    trajectory["require_perceiver_original_context"] = True
    trajectory["question_similarity_threshold"] = 0.82
    trajectory["debug_min_grounding_strict_rate"] = 0.30
    trajectory["debug_min_classification_strict_rate"] = 0.70
    geometry = trajectory.setdefault("geometry_gate", {})
    geometry["enabled"] = True
    geometry["min_hbb_iou_for_teacher_force"] = 0.10
    geometry["require_center_in_expanded_gt"] = True
    geometry["expanded_gt_ratio"] = 0.35
    geometry["pass_if_iou_or_center"] = False
    agents = result.setdefault("agents", {})
    reasoner = agents.setdefault("reasoner", {})
    reasoner["max_tokens"] = max(1024, int(reasoner.get("max_tokens", 0) or 0))
    perceiver = agents.setdefault("perceiver", {})
    perceiver["max_model_len"] = max(
        8192,
        int(perceiver.get("max_model_len", 0) or 0),
    )
    return result


def _changes(before: Any, after: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        rows: list[tuple[str, Any, Any]] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            rows.extend(_changes(before.get(key), after.get(key), path))
        return rows
    if before != after:
        return [(prefix, before, after)]
    return []


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write the upgraded settings after creating a timestamped backup.",
    )
    parser.add_argument(
        "--bump-output-version",
        action="store_true",
        help="Replace 4.3.2 with 4.3.3 in output/dataset/model paths to prevent mixing runs.",
    )
    args = parser.parse_args()

    if not args.settings.is_file():
        raise SystemExit(f"[FAIL] settings not found: {args.settings}")
    before = json.loads(args.settings.read_text(encoding="utf-8"))
    after = upgrade(before, args.bump_output_version)
    changes = _changes(before, after)
    print(f"[V4.3.3 SETTINGS] file={args.settings}")
    for key, old, new in changes:
        print(f"  {key}: {old!r} -> {new!r}")
    if not changes:
        print("  no changes required")
    if not args.in_place:
        print("[CHECK ONLY] add --in-place to write these changes")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = args.settings.with_name(
        f"{args.settings.stem}.v4.3.2_backup_{timestamp}{args.settings.suffix}"
    )
    backup.write_bytes(args.settings.read_bytes())
    _atomic_write(args.settings, after)
    print(f"[PASS] backup={backup}")
    print(f"[PASS] upgraded={args.settings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
