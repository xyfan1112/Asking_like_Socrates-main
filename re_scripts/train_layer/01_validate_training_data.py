#!/usr/bin/env python3
"""Validate B1 Direct and B2 Socratic LLaMA-Factory datasets separately."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import count_optimizer_steps, load_settings, settings_from_cli, write_json  # noqa: E402


def validate_file(path: Path) -> tuple[list[dict[str, Any]], list[str], int, dict[str, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return [], ["root_must_be_list"], 0, {}
    issues: list[str] = []
    socratic = 0
    tasks: dict[str, int] = {}
    for index, row in enumerate(rows):
        messages = row.get("messages") or []
        images = row.get("images") or []
        if len(messages) < 2 or not images:
            issues.append(f"{index}:missing_messages_or_images")
            continue
        if not Path(images[0]).is_file():
            issues.append(f"{index}:missing_image:{images[0]}")
        assistant = str(messages[-1].get("content", ""))
        metadata = row.get("metadata") or {}
        task = str(metadata.get("task") or "unknown")
        tasks[task] = tasks.get(task, 0) + 1
        is_socratic = bool(metadata.get("official_socraticagent")) or "trajectory_bucket" in metadata
        if is_socratic:
            socratic += 1
            if "<think>" in assistant:
                issues.append(f"{index}:opening_think_must_not_be_in_data_for_qwen2_vl_thinking")
            if "</think>" not in assistant:
                issues.append(f"{index}:missing_closing_think")
            if metadata.get("teacher_forced_final") is not True:
                issues.append(f"{index}:socratic_final_not_teacher_forced")
        else:
            if task == "detection_obb_by_class" and (
                "FINAL_DETECTIONS" not in assistant or "END_DETECTIONS" not in assistant
            ):
                issues.append(f"{index}:direct_detection_markers_missing")
            if task in {"ref_grounding_obb", "ref_classification"}:
                expected = str(metadata.get("raw_gt") or "").strip()
                final = assistant.rsplit("</think>", 1)[-1].strip()
                if final != expected:
                    issues.append(f"{index}:final_only_gt_mismatch")
    return rows, issues, socratic, tasks


def _names(root: Path, target: str) -> list[str]:
    b1 = ["dota128_b1_matched_train_official.json", "dota128_ref_val_official.json"]
    b2 = ["dota128_b2_matched_train_official.json", "dota128_ref_val_official.json"]
    if target == "b1":
        return b1
    if target == "b2":
        return b2
    return list(dict.fromkeys(b1 + b2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=None)
    parser.add_argument("--target", choices=["b1", "b2", "all"], default="all")
    args = parser.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    root = Path(settings["paths"]["dota128_llamafactory_root"])
    training = settings["training"]

    report: dict[str, Any] = {"target": args.target, "files": {}}
    failed = False
    for name in _names(root, args.target):
        path = root / name
        if not path.exists():
            canonical_raw = Path(settings["official_socratic"]["raw_output_dir"]) / "dota128_train_official.jsonl"
            report["files"][name] = {
                "missing": True,
                "reason": (
                    "official_conversion_not_run_or_interrupted"
                    if canonical_raw.is_file()
                    else "upstream_official_full_not_completed"
                ),
            }
            failed = True
            continue
        rows, issues, socratic, tasks = validate_file(path)
        is_val = "_val" in name
        steps = count_optimizer_steps(
            len(rows),
            training["epochs"],
            training["per_device_train_batch_size"],
            training["gradient_accumulation_steps"],
            training["world_size"],
        )
        report["files"][name] = {
            "samples": len(rows),
            "socratic_samples": socratic,
            "tasks": tasks,
            "estimated_optimizer_steps": steps,
            "issues": issues,
        }
        if issues:
            failed = True
        if "b2_matched_train" in name and socratic == 0:
            report["files"][name]["issues"].append("b2_matched_train_contains_no_socratic_samples")
            failed = True
        if not is_val and steps < int(training["min_optimizer_steps"]):
            report["files"][name]["issues"].append(
                f"optimizer_steps_below_minimum:{steps}<{training['min_optimizer_steps']}"
            )
            failed = True

    b1_path = root / "dota128_b1_matched_train_official.json"
    b2_path = root / "dota128_b2_matched_train_official.json"
    if b1_path.is_file() and b2_path.is_file():
        b1_rows = json.loads(b1_path.read_text(encoding="utf-8"))
        b2_rows = json.loads(b2_path.read_text(encoding="utf-8"))
        matched_issues: list[str] = []
        if len(b1_rows) != len(b2_rows):
            matched_issues.append(f"row_count_mismatch:{len(b1_rows)}!={len(b2_rows)}")

        def pair_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return {
                str((row.get("metadata") or {}).get("comparison_pair_id")): row
                for row in rows
                if (row.get("metadata") or {}).get("comparison_pair_id")
            }

        b1_pairs, b2_pairs = pair_index(b1_rows), pair_index(b2_rows)
        b1_direct = [
            row
            for row in b1_rows
            if not (row.get("metadata") or {}).get("comparison_pair_id")
        ]
        b2_direct = [
            row
            for row in b2_rows
            if not (row.get("metadata") or {}).get("comparison_pair_id")
        ]
        if b1_direct != b2_direct:
            matched_issues.append("direct_rows_not_identical")
        if set(b1_pairs) != set(b2_pairs):
            matched_issues.append(
                f"pair_id_set_mismatch:{len(set(b1_pairs) ^ set(b2_pairs))}"
            )
        for pair_id in sorted(set(b1_pairs) & set(b2_pairs)):
            left, right = b1_pairs[pair_id], b2_pairs[pair_id]
            if left.get("images") != right.get("images"):
                matched_issues.append(f"{pair_id}:image_mismatch")
            if (left.get("messages") or [{}])[0] != (right.get("messages") or [{}])[0]:
                matched_issues.append(f"{pair_id}:question_mismatch")
            left_gt = str((left.get("metadata") or {}).get("raw_gt") or "").strip()
            right_gt = str((right.get("metadata") or {}).get("raw_gt") or "").strip()
            left_final = str((left.get("messages") or [{}, {}])[-1].get("content", "")).rsplit(
                "</think>", 1
            )[-1].strip()
            right_final = str((right.get("messages") or [{}, {}])[-1].get("content", "")).rsplit(
                "</think>", 1
            )[-1].strip()
            if not (left_gt == right_gt == left_final == right_final):
                matched_issues.append(f"{pair_id}:final_gt_mismatch")
            if len(matched_issues) >= 100:
                break
        report["matched_comparison"] = {
            "b1_rows": len(b1_rows),
            "b2_rows": len(b2_rows),
            "pair_count": len(b1_pairs),
            "direct_rows": len(b1_direct),
            "issues": matched_issues,
        }
        if matched_issues:
            failed = True

    report["passed"] = not failed
    output = Path(settings["paths"]["training_run_root"]) / f"validation_report_{args.target}.json"
    write_json(output, report)
    print(f"[TRAIN DATA VALIDATE] {'PASS' if report['passed'] else 'FAIL'} target={args.target}")
    for name, item in report["files"].items():
        if item.get("missing"):
            print(f"  {name}: MISSING ({item.get('reason', 'unknown')})")
        else:
            print(
                f"  {name}: samples={item['samples']} socratic={item['socratic_samples']} "
                f"steps={item['estimated_optimizer_steps']} issues={len(item['issues'])}"
            )
    if failed and any(item.get("reason") == "official_conversion_not_run_or_interrupted" for item in report["files"].values()):
        print("  hint: run 'python main_layer/run.py resume-official-artifacts --settings <settings> train' after checking the full audit/API log")
    print(f"  report: {output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
