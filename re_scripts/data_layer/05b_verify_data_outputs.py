#!/usr/bin/env python3
"""Final gate for all data-layer artifacts before Socratic trajectory generation."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import load_settings, read_jsonl, settings_from_cli, write_json  # noqa: E402

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _grounding_gt_valid(row: dict) -> bool:
    gt = str(row.get("gt", ""))
    if "|" not in gt:
        return False
    nums = [float(x) for x in NUMBER_RE.findall(gt.split("|", 1)[1])]
    if len(nums) != 8 or not all(math.isfinite(v) for v in nums):
        return False
    target = row.get("coordinate_target")
    if target == "norm1000_obb":
        return all(0 <= v <= 1000 for v in nums)
    if target == "norm100_obb":
        return all(0 <= v <= 100 for v in nums)
    if target == "pixel_obb":
        w, h = int(row.get("image_width", 0)), int(row.get("image_height", 0))
        return all(
            0 <= nums[i] <= (w - 1 if i % 2 == 0 else h - 1)
            for i in range(8)
        )
    return False


def _direct_bad_lines(path: Path, target: str) -> tuple[int, int]:
    data = _json(path)
    bad = total = 0
    upper = 1000 if target == "norm1000_obb" else 100 if target == "norm100_obb" else None
    for sample in data:
        content = sample.get("messages", [{}, {}])[1].get("content", "")
        for line in content.splitlines():
            if line.startswith(("FINAL_DETECTIONS", "END_DETECTIONS")) or line.count("|") < 2:
                continue
            total += 1
            try:
                nums = [float(x) for x in line.rsplit("|", 1)[1].split(",")]
            except Exception:
                bad += 1
                continue
            if len(nums) != 8 or not all(math.isfinite(v) for v in nums):
                bad += 1
            elif upper is not None and any(v < 0 or v > upper for v in nums):
                bad += 1
    return bad, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    ref_root = Path(settings["paths"]["dota128_ref_root"])
    work = Path(settings["paths"]["pipeline_work_root"])
    agent_root = work / "agent_inputs"
    preview_root = work / "ref_previews"
    report = {"schema_version": "data_layer_gate_v4_3_3", "checks": {}, "critical": []}

    report_files = {
        "audit": work / "reports" / "dota128_audit.json",
        "ref_build": ref_root / "build_report.json",
        "ref_validation": ref_root / "validation_report.json",
        "preview": preview_root / "preview_report.json",
        "agent_build": agent_root / "build_report.json",
        "lineage": work / "reports" / "lineage_audit.json",
    }
    for name, path in report_files.items():
        exists = path.is_file() and path.stat().st_size > 0
        report["checks"][f"report_{name}"] = {"path": str(path), "exists": exists}
        if not exists:
            report["critical"].append(f"missing report: {path}")

    for key in ("ref_validation", "preview", "agent_build", "lineage"):
        path = report_files[key]
        if path.is_file():
            passed = bool(_json(path).get("passed", False))
            report["checks"][f"{key}_passed"] = passed
            if not passed:
                report["critical"].append(f"{key} did not pass: {path}")

    target = settings["data_conversion"].get("coordinate_target", "norm1000_obb")
    for split in settings["data_conversion"].get("splits", ["train", "val"]):
        required = {
            "ref": ref_root / f"{split}.jsonl",
            "all": ref_root / f"{split}_all.jsonl",
            "agent": agent_root / f"{split}_agent_inputs.jsonl",
            "direct": agent_root / f"{split}_direct.json",
        }
        for name, path in required.items():
            exists = path.is_file() and path.stat().st_size > 0
            report["checks"][f"{split}_{name}"] = {"path": str(path), "exists": exists}
            if not exists:
                report["critical"].append(f"missing artifact: {path}")
        agent_path = required["agent"]
        if agent_path.is_file():
            rows = read_jsonl(agent_path)
            grounding = [r for r in rows if r.get("task") == "ref_grounding_obb"]
            bad_gt = sum(not _grounding_gt_valid(r) for r in grounding)
            report["checks"][f"{split}_agent_stats"] = {
                "rows": len(rows),
                "grounding_rows": len(grounding),
                "invalid_grounding_gt": bad_gt,
            }
            if bad_gt:
                report["critical"].append(f"{split}: invalid grounding GT={bad_gt}")
        direct_path = required["direct"]
        if direct_path.is_file():
            bad_lines, total_lines = _direct_bad_lines(direct_path, target)
            report["checks"][f"{split}_direct_stats"] = {
                "object_lines": total_lines,
                "invalid_object_lines": bad_lines,
            }
            if bad_lines:
                report["critical"].append(f"{split}: invalid direct OBB lines={bad_lines}")

    report["passed"] = not report["critical"]
    report["next_step"] = (
        "python main_layer/run.py build-official-parquet --settings settings.json --split train"
        if report["passed"]
        else "fix the first critical item, then rerun data-full"
    )
    out = work / "reports" / "data_layer_final_report.json"
    write_json(out, report)

    print(f"[DATA LAYER GATE] {'PASS' if report['passed'] else 'FAIL'}")
    for split in settings["data_conversion"].get("splits", ["train", "val"]):
        a = report["checks"].get(f"{split}_agent_stats", {})
        d = report["checks"].get(f"{split}_direct_stats", {})
        print(
            f"  {split}: agent_rows={a.get('rows', 0)} invalid_grounding_gt={a.get('invalid_grounding_gt', 0)} "
            f"direct_objects={d.get('object_lines', 0)} invalid_direct={d.get('invalid_object_lines', 0)}"
        )
    if report["critical"]:
        print("  critical:")
        for item in report["critical"]:
            print(f"    - {item}")
    print(f"  report: {out}")
    print(f"  next: {report['next_step']}")
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
