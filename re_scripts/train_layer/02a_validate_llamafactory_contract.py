#!/usr/bin/env python3
"""Validate the exact LLaMA-Factory dataset_dir/dataset_info contract before GPU launch."""
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


TARGET_DATASETS = {
    "b1": ["dota128_b1_matched_train_official", "dota128_ref_val_official"],
    "b2": ["dota128_b2_matched_train_official", "dota128_ref_val_official"],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_image_tags(messages: list[dict[str, Any]]) -> int:
    return sum(str(m.get("content", "")).count("<image>") for m in messages)


def validate_dataset_file(path: Path, max_issues: int = 50) -> tuple[int, list[str]]:
    issues: list[str] = []
    try:
        rows = load_json(path)
    except Exception as exc:
        return 0, [f"json_load_failed:{type(exc).__name__}:{exc}"]
    if not isinstance(rows, list):
        return 0, ["root_must_be_list"]
    if not rows:
        return 0, ["dataset_empty"]

    for index, row in enumerate(rows):
        if len(issues) >= max_issues:
            break
        if not isinstance(row, dict):
            issues.append(f"{index}:row_not_object")
            continue
        messages = row.get("messages")
        images = row.get("images")
        if not isinstance(messages, list) or len(messages) < 2:
            issues.append(f"{index}:messages_invalid")
            continue
        if not isinstance(images, list) or not images:
            issues.append(f"{index}:images_invalid")
            continue
        tag_count = count_image_tags(messages)
        if tag_count != len(images):
            issues.append(f"{index}:image_tag_count_mismatch:{tag_count}!={len(images)}")
        for image in images:
            if not Path(str(image)).is_file():
                issues.append(f"{index}:missing_image:{image}")
                if len(issues) >= max_issues:
                    break
        if str(messages[-1].get("role", "")) != "assistant":
            issues.append(f"{index}:last_message_not_assistant")
        if not str(messages[-1].get("content", "")).strip():
            issues.append(f"{index}:assistant_content_empty")
    return len(rows), issues


def parse_yaml_minimal(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {"dataset_dir", "dataset", "eval_dataset", "output_dir", "template"}:
            values[key.strip()] = value.strip().strip('"\'')
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--target", choices=["b1", "b2", "all"], default="b1")
    ap.add_argument("--config", help="Optional generated YAML path; inferred when omitted.")
    args = ap.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    target_names = ["b1", "b2"] if args.target == "all" else [args.target]
    dataset_dir = Path(settings["paths"]["dota128_llamafactory_root"]).resolve()
    info_path = dataset_dir / "dataset_info.json"
    report: dict[str, Any] = {
        "target": args.target,
        "dataset_dir": str(dataset_dir),
        "dataset_info": str(info_path),
        "checks": {},
        "critical": [],
    }

    if not dataset_dir.is_dir():
        report["critical"].append(f"dataset_dir_missing:{dataset_dir}")
        info: dict[str, Any] = {}
    elif not info_path.is_file():
        report["critical"].append(f"dataset_info_missing:{info_path}")
        info = {}
    else:
        try:
            info = load_json(info_path)
            if not isinstance(info, dict):
                raise TypeError("root must be object")
        except Exception as exc:
            report["critical"].append(f"dataset_info_invalid:{type(exc).__name__}:{exc}")
            info = {}

    required = []
    for target in target_names:
        required.extend(TARGET_DATASETS[target])
    required = list(dict.fromkeys(required))

    for name in required:
        item: dict[str, Any] = {}
        entry = info.get(name)
        if not isinstance(entry, dict):
            item["issues"] = ["undefined_in_dataset_info"]
            report["critical"].append(f"undefined_dataset:{name}")
            report["checks"][name] = item
            continue
        file_name = entry.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            item["issues"] = ["file_name_missing"]
            report["critical"].append(f"file_name_missing:{name}")
            report["checks"][name] = item
            continue
        path = Path(file_name)
        if not path.is_absolute():
            path = dataset_dir / path
        path = path.resolve()
        item["file_name"] = file_name
        item["resolved_path"] = str(path)
        item["formatting"] = entry.get("formatting")
        item["columns"] = entry.get("columns")
        item["tags"] = entry.get("tags")
        issues: list[str] = []
        if entry.get("formatting") != "sharegpt":
            issues.append("formatting_must_be_sharegpt")
        columns = entry.get("columns") or {}
        if columns.get("messages") != "messages" or columns.get("images") != "images":
            issues.append("columns_messages_images_mismatch")
        tags = entry.get("tags") or {}
        for key, expected in {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        }.items():
            if tags.get(key) != expected:
                issues.append(f"tag_mismatch:{key}")
        if not path.is_file():
            issues.append("dataset_file_missing")
            rows = 0
        else:
            rows, file_issues = validate_dataset_file(path)
            issues.extend(file_issues)
        item["rows"] = rows
        item["issues"] = issues
        report["checks"][name] = item
        if issues:
            report["critical"].append(f"dataset_invalid:{name}")

    config_root = Path(settings["paths"]["training_run_root"]) / "configs"
    inferred = {
        "b1": config_root / "b1_direct_lora.yaml",
        "b2": config_root / "b2_socratic_lora.yaml",
    }
    yaml_reports: dict[str, Any] = {}
    for target in target_names:
        config_path = Path(args.config) if args.config and len(target_names) == 1 else inferred[target]
        values = parse_yaml_minimal(config_path)
        item = {"path": str(config_path), "exists": config_path.is_file(), "values": values, "issues": []}
        if not config_path.is_file():
            item["issues"].append("config_missing")
        else:
            configured_dir = Path(values.get("dataset_dir", "")).resolve() if values.get("dataset_dir") else None
            if configured_dir != dataset_dir:
                item["issues"].append(f"dataset_dir_mismatch:{configured_dir}!={dataset_dir}")
            expected_train, expected_eval = TARGET_DATASETS[target]
            if values.get("dataset") != expected_train:
                item["issues"].append(f"dataset_name_mismatch:{values.get('dataset')}!={expected_train}")
            if values.get("eval_dataset") != expected_eval:
                item["issues"].append(f"eval_dataset_mismatch:{values.get('eval_dataset')}!={expected_eval}")
        yaml_reports[target] = item
        if item["issues"]:
            report["critical"].append(f"yaml_invalid:{target}")
    report["yaml"] = yaml_reports

    # Use the exact parser from the user's LLaMA-Factory checkout when available.
    parser_check: dict[str, Any] = {"attempted": False}
    try:
        lf_root = resolve_llama_factory_dir(settings, settings["training"]["base_model_key"])
        src = lf_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from llamafactory.data.parser import get_dataset_list  # type: ignore

        parser_check["attempted"] = True
        attrs = get_dataset_list(required, str(dataset_dir))
        parser_check["resolved"] = [getattr(x, "dataset_name", None) for x in attrs]
        parser_check["passed"] = len(attrs) == len(required)
        if not parser_check["passed"]:
            report["critical"].append("llamafactory_parser_count_mismatch")
    except Exception as exc:
        parser_check["attempted"] = True
        parser_check["passed"] = False
        parser_check["error"] = f"{type(exc).__name__}: {exc}"
        report["critical"].append("llamafactory_parser_failed")
    report["llamafactory_parser"] = parser_check

    report["passed"] = not report["critical"]
    output = Path(settings["paths"]["training_run_root"]) / f"llamafactory_contract_{args.target}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)

    print(f"[LLAMAFACTORY CONTRACT] {'PASS' if report['passed'] else 'FAIL'} target={args.target}")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  dataset_info: {info_path}")
    for name in required:
        item = report["checks"].get(name, {})
        print(f"  {name}: rows={item.get('rows', 0)} issues={len(item.get('issues', []))}")
    for target, item in yaml_reports.items():
        print(f"  yaml {target}: {item['path']} issues={len(item['issues'])}")
    print(f"  parser_check: {'PASS' if parser_check.get('passed') else 'FAIL'}")
    print(f"  report: {output}")
    if not report["passed"]:
        print("  critical:")
        for issue in report["critical"]:
            print(f"    - {issue}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
