#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import parse_obb_output_status, split_center_rois  # noqa: E402


def test_entry_points() -> None:
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "main_layer/run.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    for command in ("split-scenes", "stop-eval", "audit-lineage", "compare-b0-b1-b2"):
        assert command in help_text


def test_parser() -> None:
    bad = parse_obb_output_status(
        "<think>ship 10 10 20 10 20 20 10 20</think>\nship maybe",
        1000,
        1000,
        "normalized_0_1000",
        strict=True,
        require_markers=True,
        expected_class="ship",
        require_confidence=True,
    )
    assert bad["predictions"] == []
    good = parse_obb_output_status(
        "FINAL_DETECTIONS\nship|0.8|10,10,20,10,20,20,10,20\nEND_DETECTIONS",
        1000,
        1000,
        "normalized_0_1000",
        strict=True,
        require_markers=True,
        expected_class="ship",
        require_confidence=True,
    )
    assert good["format_ok"] and len(good["predictions"]) == 1


def test_rois() -> None:
    rows = [
        {"center_pixel": [float(index * 10), float(index)], "object_index": index}
        for index in range(95)
    ]
    groups = split_center_rois(rows, 40, 1000, 1000)
    assert len(groups) >= 3
    assert all(len(group) <= 40 for _, group in groups)
    assert sorted(item["object_index"] for _, group in groups for item in group) == list(
        range(95)
    )


def test_metrics() -> None:
    with tempfile.TemporaryDirectory(prefix="v432_metrics_") as td:
        temp = Path(td)
        settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
        settings["paths"]["test_run_root"] = str(temp)
        settings_path = temp / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        points = [[10, 10], [20, 10], [20, 20], [10, 20]]
        row = {
            "query_id": "q1",
            "run_id": 0,
            "ground_truth": [{"class_id": 1, "class_name": "ship", "points": points}],
            "predictions": [
                {
                    "class_id": 1,
                    "class_name": "ship",
                    "points": points,
                    "confidence": 1.0,
                }
            ],
            "format_ok": True,
            "parse_ok": True,
            "error": None,
            "finish_reason": "stop",
        }
        pred_path = temp / "det.jsonl"
        pred_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        pred_path.with_suffix(".manifest.json").write_text(
            json.dumps({"expected_rows": 1}), encoding="utf-8"
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "test_layer/02_raw_dota_obb_metrics.py"),
                "--settings",
                str(settings_path),
                "--pred",
                str(pred_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        result = json.loads(
            (temp / "det_metrics.json").read_text(encoding="utf-8")
        )["summary"]
        assert result["status"] == "VALID"
        assert abs(result["mean_iou_per_gt"] - 1.0) < 1e-9
        assert abs(result["recall_iou_0_5"] - 1.0) < 1e-9


def main() -> None:
    test_entry_points()
    test_parser()
    test_rois()
    test_metrics()
    print("V4.3.3 REGRESSION TESTS PASS")


if __name__ == "__main__":
    main()
