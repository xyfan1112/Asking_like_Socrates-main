#!/usr/bin/env python3
"""Filter official raw trajectories using deterministic v4.3.3 gates.

The official LLM verifier is recorded as advisory. It is not allowed to discard a
Grounding trace merely because approximate coordinates differ from the exact GT;
target identity/trace consistency and deterministic coarse geometry are the
actual gates before GT teacher forcing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main_layer"))
from main_layer.common import load_settings, read_jsonl, settings_from_cli, write_json, write_jsonl  # noqa: E402
from data_layer.trajectory_gates import (  # noqa: E402
    audit_generated_answer,
    audit_trace_semantics,
    audit_trajectory_evidence,
)

HARD_LOOP_ERRORS = {
    "reasoner_missing_thinking",
    "output_format_error",
    "max_rounds_exceeded",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    official = settings["official_socratic"]
    raw_path = Path(args.input) if args.input else Path(official["raw_output_dir"]) / f"dota128_{args.split}_official.jsonl"
    filtered_path = Path(args.output) if args.output else Path(official["raw_output_dir"]) / f"dota128_{args.split}_official_strict.jsonl"
    agent_path = Path(settings["paths"]["pipeline_work_root"]) / "agent_inputs" / f"{args.split}_agent_inputs.jsonl"
    index = {row["id"]: row for row in read_jsonl(agent_path)}
    raw_rows = read_jsonl(raw_path, skip_bad=True)

    accepted: list[dict] = []
    audit_rows: list[dict] = []
    reasons: Counter[str] = Counter()
    accepted_by_task: Counter[str] = Counter()
    input_by_task: Counter[str] = Counter()
    semantic_metrics: Counter[str] = Counter()
    grounding_gate_metrics: Counter[str] = Counter()
    min_rounds_default = int(settings["trajectory"].get("min_perception_rounds_strict", 2))
    min_rounds_by_task = settings["trajectory"].get(
        "min_perception_rounds_by_task",
        {"ref_grounding_obb": 2, "ref_classification": 1},
    )

    for raw in raw_rows:
        rid = str(raw.get("id", ""))
        item = index.get(rid)
        task = (item or raw).get("task", "unknown")
        input_by_task[task] += 1
        loop = raw.get("loop_result") if isinstance(raw.get("loop_result"), dict) else {}
        final = str(loop.get("final_answer") or "")
        history = loop.get("chat_history") if isinstance(loop.get("chat_history"), list) else []
        rounds = sum(1 for turn in history if turn.get("P_response"))
        loop_error = str(loop.get("error") or raw.get("error") or "").strip()
        official_success = bool(loop.get("success"))
        official_reasons = [
            str(turn.get("V_reason"))
            for turn in history
            if turn.get("V_decision") == "REJECT" and turn.get("V_reason")
        ]

        why: list[str] = []
        if not item:
            why.append("missing_agent_input")
            geometry = {"pass": False, "reason": "missing_agent_input"}
            semantic = {"pass": False, "reasons": ["missing_agent_input"]}
        else:
            if task == "ref_grounding_obb":
                geometry = audit_trajectory_evidence(item, raw, settings)
            else:
                geometry = audit_generated_answer(item, final, settings)
            semantic = audit_trace_semantics(item, raw, settings)
            if semantic.get("duplicate_question_trajectory"):
                semantic_metrics["duplicate_question_trajectories"] += 1
            if task == "ref_classification" and semantic.get(
                "classification_contradiction"
            ):
                semantic_metrics["classification_contradiction_trajectories"] += 1
            if task == "ref_classification" and semantic.get(
                "classification_leading_questions"
            ):
                semantic_metrics["classification_leading_question_trajectories"] += 1
            if task == "ref_grounding_obb":
                grounding_gate_metrics["iou_pass"] += int(bool(geometry.get("iou_gate")))
                grounding_gate_metrics["center_pass"] += int(
                    bool(geometry.get("center_gate"))
                )
                grounding_gate_metrics["both_pass"] += int(
                    bool(geometry.get("iou_gate"))
                    and bool(geometry.get("center_gate"))
                )

        if loop_error in HARD_LOOP_ERRORS:
            why.append(f"official_loop:{loop_error}")
        elif loop_error and loop_error not in {"unknown_error", "unknown"}:
            why.append(f"official_loop:{loop_error}")

        required_rounds = int(min_rounds_by_task.get(task, min_rounds_default))
        if rounds < required_rounds:
            why.append(f"too_few_perception_rounds:{rounds}<{required_rounds}")
        if not geometry.get("pass"):
            why.append(f"geometry:{geometry.get('reason')}")
        if not semantic.get("pass"):
            for reason in semantic.get("reasons") or ["semantic_inconsistent"]:
                why.append(f"semantic:{reason}")

        passed = not why
        if passed:
            accepted.append(raw)
            accepted_by_task[task] += 1
        else:
            for reason in why:
                reasons[reason] += 1

        audit_rows.append(
            {
                "id": rid,
                "task": task,
                "passed": passed,
                "reasons": why,
                "perception_rounds": rounds,
                "generated_final": final,
                "gt": item.get("gt") if item else raw.get("gt"),
                "geometry_audit": geometry,
                "semantic_audit": semantic,
                "official_verifier_advisory": {
                    "success": official_success,
                    "reject_reasons": official_reasons,
                },
            }
        )

    write_jsonl(filtered_path, accepted)
    audit_path = filtered_path.with_suffix(".audit.json")
    report_path = filtered_path.with_suffix(".report.json")
    write_json(audit_path, audit_rows)
    report = {
        "schema_version": "official_strict_v4_3_3",
        "raw_input": str(raw_path),
        "agent_input": str(agent_path),
        "strict_output": str(filtered_path),
        "input_rows": len(raw_rows),
        "input_by_task": dict(input_by_task),
        "accepted_strict": len(accepted),
        "accepted_by_task": dict(accepted_by_task),
        "rejected": len(raw_rows) - len(accepted),
        "rejection_reasons": dict(reasons.most_common()),
        "official_verifier_is_advisory": True,
        "teacher_force_final_answer": bool(settings["trajectory"].get("teacher_force_final_answer", True)),
        "geometry_gate": settings["trajectory"].get("geometry_gate", {}),
        "semantic_quality_counts": dict(semantic_metrics),
        "grounding_gate_counts": dict(grounding_gate_metrics),
    }
    write_json(report_path, report)
    print("[OFFICIAL STRICT AUDIT] {}".format("PASS" if accepted else "FAIL"))
    print(f"  input={len(raw_rows)} accepted={len(accepted)} rejected={len(raw_rows)-len(accepted)}")
    print(f"  input_by_task: {dict(input_by_task)}")
    print(f"  accepted_by_task: {dict(accepted_by_task)}")
    print(f"  semantic_quality: {dict(semantic_metrics)}")
    print(f"  grounding_gates: {dict(grounding_gate_metrics)}")
    print(f"  top_rejections: {dict(reasons.most_common(10))}")
    print(f"  strict: {filtered_path}")
    print(f"  report: {report_path}")
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
