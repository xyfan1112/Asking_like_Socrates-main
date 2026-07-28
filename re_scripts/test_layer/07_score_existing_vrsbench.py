#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--pred")
    ap.add_argument("--k", type=int)
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    pred = args.pred or settings["paths"]["vrsbench_result_jsonl"]
    k = args.k or int(settings["evaluation"]["protocols"]["vrsbench_vqa_k5"]["k"])
    output = Path(settings["paths"]["test_run_root"]) / "vrsbench_existing_score.json"
    command = [
        sys.executable,
        str(ROOT / "test_layer/legacy_vrsbench/03_score_local.py"),
        "--pred", str(pred), "--k", str(k), "--save-summary", str(output),
    ]
    print("+", " ".join(command))
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
