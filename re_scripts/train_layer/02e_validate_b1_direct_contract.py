#!/usr/bin/env python3
"""Validate dataset_info, files, YAML and the user's LLaMA-Factory parser."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import load_settings, settings_from_cli, write_json  # noqa: E402
from config import resolve_llama_factory_dir  # noqa: E402

DATASETS = ["dota128_direct_train_official", "dota128_direct_val_official"]


def parse_yaml(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {"dataset_dir", "dataset", "eval_dataset", "output_dir", "template"}:
            out[key.strip()] = value.strip().strip("\"'")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    dataset_dir = Path(settings["paths"]["dota128_llamafactory_root"]).resolve()
    info_path = dataset_dir / "dataset_info.json"
    cfg_path = Path(settings["paths"]["training_run_root"]) / "configs" / "b1_direct_standalone_lora.yaml"
    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "dataset_info": str(info_path),
        "config": str(cfg_path),
        "critical": [],
        "datasets": {},
    }
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise TypeError("dataset_info root must be object")
    except Exception as exc:
        info = {}
        report["critical"].append(f"dataset_info_invalid:{type(exc).__name__}:{exc}")
    for name in DATASETS:
        entry = info.get(name)
        item: dict[str, Any] = {"issues": []}
        if not isinstance(entry, dict):
            item["issues"].append("undefined_in_dataset_info")
        else:
            file_name = entry.get("file_name")
            path = Path(str(file_name))
            if not path.is_absolute():
                path = dataset_dir / path
            item["path"] = str(path)
            if not path.is_file():
                item["issues"].append("dataset_file_missing")
            else:
                try:
                    rows = json.loads(path.read_text(encoding="utf-8"))
                    item["rows"] = len(rows) if isinstance(rows, list) else 0
                    if not isinstance(rows, list) or not rows:
                        item["issues"].append("dataset_empty_or_not_list")
                except Exception as exc:
                    item["issues"].append(f"json_invalid:{type(exc).__name__}:{exc}")
            if entry.get("formatting") != "sharegpt":
                item["issues"].append("formatting_not_sharegpt")
        if item["issues"]:
            report["critical"].append(f"dataset_invalid:{name}")
        report["datasets"][name] = item
    values = parse_yaml(cfg_path)
    report["yaml"] = values
    if not cfg_path.is_file():
        report["critical"].append("training_yaml_missing")
    else:
        if Path(values.get("dataset_dir", "")).resolve() != dataset_dir:
            report["critical"].append("yaml_dataset_dir_mismatch")
        if values.get("dataset") != DATASETS[0]:
            report["critical"].append("yaml_train_dataset_mismatch")
        if values.get("eval_dataset") != DATASETS[1]:
            report["critical"].append("yaml_eval_dataset_mismatch")
    parser_report: dict[str, Any] = {"attempted": False}
    try:
        lf_root = resolve_llama_factory_dir(settings, settings["training"]["base_model_key"])
        src = lf_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from llamafactory.data.parser import get_dataset_list  # type: ignore
        parser_report["attempted"] = True
        attrs = get_dataset_list(DATASETS, str(dataset_dir))
        parser_report["resolved"] = [getattr(item, "dataset_name", None) for item in attrs]
        parser_report["passed"] = len(attrs) == len(DATASETS)
        if not parser_report["passed"]:
            report["critical"].append("llamafactory_parser_count_mismatch")
    except Exception as exc:
        parser_report.update(
            attempted=True,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        report["critical"].append("llamafactory_parser_failed")
    report["llamafactory_parser"] = parser_report
    report["passed"] = not report["critical"]
    output = Path(settings["paths"]["training_run_root"]) / "b1_direct_standalone_contract.json"
    write_json(output, report)
    print(f"[B1 DIRECT CONTRACT] {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  dataset_dir={dataset_dir}")
    print(f"  config={cfg_path}")
    print(f"  parser={'PASS' if parser_report.get('passed') else 'FAIL'}")
    print(f"  report={output}")
    for issue in report["critical"]:
        print(f"  critical: {issue}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
