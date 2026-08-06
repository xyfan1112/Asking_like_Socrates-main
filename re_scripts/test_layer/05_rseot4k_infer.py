#!/usr/bin/env python3
"""Compatibility wrapper for the old RS-EoT-4K command.

New experiments should call 05_sharegpt_dataset_infer.py directly. This wrapper
keeps the old filename but now validates the configured dataset path and uses the
separate VQA protocol instead of obsolete flat evaluation keys.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--model-key", default="rs_eot")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--k", type=int)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()
    settings_path = settings_from_cli(__file__, args.settings)
    settings = load_settings(settings_path)
    dataset = Path(settings["paths"]["rs_eot4k_llamafactory"])
    if not dataset.is_file():
        raise FileNotFoundError(
            f"Configured RS-EoT-4K ShareGPT file does not exist: {dataset}. "
            "Update paths.rs_eot4k_llamafactory or call 05_sharegpt_dataset_infer.py with --dataset-path."
        )
    command = [
        sys.executable, str(ROOT / "test_layer/05_sharegpt_dataset_infer.py"),
        "--settings", str(settings_path), "--dataset-path", str(dataset),
        "--model-key", args.model_key, "--base-url", args.base_url,
        "--protocol", "vrsbench_vqa_k5",
    ]
    if args.k is not None:
        command += ["--k", str(args.k)]
    if args.max_samples:
        command += ["--max-samples", str(args.max_samples)]
    if args.output:
        command += ["--output", args.output]
    print("[DEPRECATED] 05_rseot4k_infer.py delegates to 05_sharegpt_dataset_infer.py")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
