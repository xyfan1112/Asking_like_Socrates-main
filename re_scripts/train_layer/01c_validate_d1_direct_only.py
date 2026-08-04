#!/usr/bin/env python3
"""Validate deterministic Direct-only D1 data without requiring Socratic output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import count_optimizer_steps, load_settings, settings_from_cli, write_json  # noqa: E402


def validate(path: Path) -> tuple[int, list[str]]:
    if not path.is_file():
        return 0, [f"missing:{path}"]
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return 0, ["root_not_nonempty_list"]
    issues: list[str] = []
    for index, row in enumerate(rows):
        messages = row.get("messages") or []
        images = row.get("images") or []
        if len(messages) < 2 or not images:
            issues.append(f"{index}:missing_messages_or_images")
            continue
        if not Path(str(images[0])).is_file():
            issues.append(f"{index}:missing_image:{images[0]}")
        assistant = str(messages[-1].get("content") or "")
        if "FINAL_DETECTIONS" not in assistant or "END_DETECTIONS" not in assistant:
            issues.append(f"{index}:direct_detection_markers_missing")
        metadata = row.get("metadata") or {}
        if metadata.get("official_socraticagent") or "trajectory_bucket" in metadata:
            issues.append(f"{index}:socratic_row_not_allowed_in_d1")
    return len(rows), issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    root = Path(settings["paths"]["dota128_llamafactory_root"])
    files = {
        "train": root / "dota128_direct_train_official.json",
        "val": root / "dota128_direct_val_official.json",
    }
    report = {"mode": "d1_direct_only", "files": {}}
    failed = False
    for split, path in files.items():
        count, issues = validate(path)
        report["files"][split] = {"path": str(path), "samples": count, "issues": issues[:200]}
        failed = failed or bool(issues)
    training = settings["training"]
    report["estimated_optimizer_steps"] = count_optimizer_steps(
        report["files"]["train"]["samples"],
        training["epochs"],
        training["per_device_train_batch_size"],
        training["gradient_accumulation_steps"],
        training["world_size"],
    )
    report_path = root / "d1_direct_only_validation.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        print("[D1 VALIDATE] FAIL")
        return 2
    print("[D1 VALIDATE] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
