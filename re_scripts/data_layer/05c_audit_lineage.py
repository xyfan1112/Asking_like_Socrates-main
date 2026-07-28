#!/usr/bin/env python3
"""Audit OBB/label lineage across DOTA -> Ref -> agent/Direct -> matched SFT."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import (  # noqa: E402
    coordinate_points,
    load_settings,
    obb_iou,
    read_jsonl,
    settings_from_cli,
    write_json,
)


def final_text(row: dict) -> str:
    messages = row.get("messages") or []
    return str(messages[-1].get("content", "")) if messages else ""


def audit_sft_pairs(root: Path) -> dict:
    b1_path = root / "dota128_b1_matched_train_official.json"
    b2_path = root / "dota128_b2_matched_train_official.json"
    if not b1_path.is_file() or not b2_path.is_file():
        return {"available": False, "issues": []}
    b1 = json.loads(b1_path.read_text(encoding="utf-8"))
    b2 = json.loads(b2_path.read_text(encoding="utf-8"))

    def pairs(rows: list[dict]) -> dict[str, dict]:
        return {
            str((row.get("metadata") or {}).get("comparison_pair_id")): row
            for row in rows
            if (row.get("metadata") or {}).get("comparison_pair_id")
        }

    left, right = pairs(b1), pairs(b2)
    issues = []
    if len(b1) != len(b2):
        issues.append(f"row_count_mismatch:{len(b1)}!={len(b2)}")
    if set(left) != set(right):
        issues.append(f"pair_set_mismatch:{len(set(left) ^ set(right))}")
    for pair_id in sorted(set(left) & set(right)):
        a, b = left[pair_id], right[pair_id]
        a_gt = str((a.get("metadata") or {}).get("raw_gt") or "").strip()
        b_gt = str((b.get("metadata") or {}).get("raw_gt") or "").strip()
        a_final = final_text(a).rsplit("</think>", 1)[-1].strip()
        b_final = final_text(b).rsplit("</think>", 1)[-1].strip()
        if a.get("images") != b.get("images"):
            issues.append(f"{pair_id}:image")
        if (a.get("messages") or [{}])[0] != (b.get("messages") or [{}])[0]:
            issues.append(f"{pair_id}:question")
        if not (a_gt == b_gt == a_final == b_final):
            issues.append(f"{pair_id}:final")
        if len(issues) >= 100:
            break
    return {
        "available": True,
        "b1_rows": len(b1),
        "b2_rows": len(b2),
        "pair_count": len(left),
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    ref_root = Path(settings["paths"]["dota128_ref_root"])
    agent_root = Path(settings["paths"]["pipeline_work_root"]) / "agent_inputs"
    lf_root = Path(settings["paths"]["dota128_llamafactory_root"])
    critical: list[str] = []
    splits = {}

    for split in settings["data_conversion"].get("splits", ["train", "val"]):
        all_rows = read_jsonl(ref_root / f"{split}_all.jsonl")
        refs = read_jsonl(ref_root / f"{split}.jsonl")
        all_index = {str(row["id"]): row for row in all_rows}
        if len(all_index) != len(all_rows):
            critical.append(f"{split}:duplicate_ref_ids")
        unknown_selected = [row["id"] for row in refs if str(row["id"]) not in all_index]
        if unknown_selected:
            critical.append(f"{split}:selected_not_in_all:{len(unknown_selected)}")

        agent_rows = read_jsonl(agent_root / f"{split}_agent_inputs.jsonl")
        agent_issues = 0
        max_iou_error = 0.0
        task_counts: Counter[str] = Counter()
        for item in agent_rows:
            source_id = str(item.get("source_ref_id") or "")
            source = all_index.get(source_id)
            if source is None:
                agent_issues += 1
                continue
            task_counts[str(item.get("task"))] += 1
            if item.get("class_name") != source.get("class_name"):
                agent_issues += 1
            score = obb_iou(item.get("obb_pixel", []), source.get("obb_pixel", []))
            max_iou_error = max(max_iou_error, 1.0 - score)
            if score < 0.999999:
                agent_issues += 1
        if agent_issues:
            critical.append(f"{split}:agent_lineage_issues:{agent_issues}")

        direct_rows = json.loads(
            (agent_root / f"{split}_direct.json").read_text(encoding="utf-8")
        )
        coverage: Counter[str] = Counter()
        direct_issues = 0
        for sample in direct_rows:
            metadata = sample.get("metadata") or {}
            source_ids = [str(value) for value in metadata.get("source_ref_ids", [])]
            for source_id in source_ids:
                coverage[source_id] += 1
                source = all_index.get(source_id)
                if source is None:
                    direct_issues += 1
                    continue
                if source.get("class_name") != metadata.get("class_name"):
                    direct_issues += 1
                roi = metadata.get("roi_pixel") or []
                cx, cy = source["center_pixel"]
                if len(roi) != 4 or not (
                    float(roi[0]) <= cx <= float(roi[2])
                    and float(roi[1]) <= cy <= float(roi[3])
                ):
                    direct_issues += 1
            if bool(metadata.get("negative_sample")) != (len(source_ids) == 0):
                direct_issues += 1

            answer = final_text(sample)
            answer_lines = [
                line.strip()
                for line in answer.splitlines()
                if "|" in line and "FINAL_DETECTIONS" not in line
            ]
            if len(answer_lines) != len(source_ids):
                direct_issues += 1
            for line, source_id in zip(answer_lines, source_ids):
                source = all_index.get(source_id)
                if source is None:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    direct_issues += 1
                    continue
                actual = [int(float(value)) for value in parts[2].split(",")]
                expected = [
                    int(round(value))
                    for point in coordinate_points(
                        source["obb_pixel"],
                        int(source["image_width"]),
                        int(source["image_height"]),
                        str(metadata.get("coordinate_target")),
                    )
                    for value in point
                ]
                if actual != expected:
                    direct_issues += 1

        missing_direct = set(all_index) - set(coverage)
        duplicate_direct = sum(value - 1 for value in coverage.values() if value > 1)
        if missing_direct:
            critical.append(f"{split}:direct_missing_objects:{len(missing_direct)}")
        if duplicate_direct:
            critical.append(f"{split}:direct_duplicate_objects:{duplicate_direct}")
        if direct_issues:
            critical.append(f"{split}:direct_lineage_issues:{direct_issues}")

        splits[split] = {
            "all_objects": len(all_rows),
            "ref_rows": len(refs),
            "agent_rows": len(agent_rows),
            "agent_task_counts": dict(task_counts),
            "agent_max_obb_iou_error": max_iou_error,
            "direct_samples": len(direct_rows),
            "direct_unique_objects": len(coverage),
            "direct_missing_objects": len(missing_direct),
            "direct_duplicate_objects": duplicate_direct,
        }

    sft = audit_sft_pairs(lf_root)
    if sft.get("issues"):
        critical.append(f"sft_matched_pair_issues:{len(sft['issues'])}")
    report = {
        "schema_version": "lineage_audit_v1",
        "splits": splits,
        "sft_matched_comparison": sft,
        "critical": critical,
        "passed": not critical,
    }
    output = Path(settings["paths"]["pipeline_work_root"]) / "reports" / "lineage_audit.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[LINEAGE] {'PASS' if report['passed'] else 'FAIL'} report={output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
