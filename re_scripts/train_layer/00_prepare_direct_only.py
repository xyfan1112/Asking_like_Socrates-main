#!/usr/bin/env python3
"""Prepare B1 Direct OBB datasets and a self-contained LLaMA-Factory registry.

Authoritative layout:
    <dota128_llamafactory_root>/
      dataset_info.json
      dota128_direct_train_official.json
      dota128_direct_val_official.json

LLaMA-Factory always loads ``dataset_info.json`` from ``dataset_dir``.  The
training YAML therefore points to this directory.  ``--register`` additionally
mirrors the entries into ``LLaMA-Factory/data/dataset_info.json`` for WebUI
convenience, but training does not depend on that mirror.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import load_settings, settings_from_cli, write_json  # noqa: E402
from config import resolve_llama_factory_dir  # noqa: E402


DATASET_NAMES = {
    "train": "dota128_direct_train_official",
    "val": "dota128_direct_val_official",
}


def dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"dataset_info.json root must be an object: {path}")
    return value


def update_registry(path: Path, entries: dict[str, dict[str, Any]], backup_suffix: str | None = None) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if path.exists() and backup_suffix:
        backup = path.with_name(path.name + backup_suffix)
        if not backup.exists():
            shutil.copyfile(path, backup)
    info = load_registry(path)
    info.update(entries)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return backup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Also mirror the dataset entries into LLaMA-Factory/data/dataset_info.json for WebUI use.",
    )
    args = parser.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))

    agent_dir = Path(settings["paths"]["pipeline_work_root"]) / "agent_inputs"
    dataset_dir = Path(settings["paths"]["dota128_llamafactory_root"])
    dataset_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    counts: dict[str, int] = {}
    entries_local: dict[str, dict[str, Any]] = {}
    entries_official: dict[str, dict[str, Any]] = {}

    for split in ("train", "val"):
        src = agent_dir / f"{split}_direct.json"
        if not src.is_file():
            raise FileNotFoundError(
                f"Missing {src}. Run: python main_layer/run.py data-full --settings settings.json"
            )
        rows = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Direct dataset is empty or invalid: {src}")

        dst = dataset_dir / f"dota128_direct_{split}_official.json"
        shutil.copyfile(src, dst)
        outputs[split] = str(dst)
        counts[split] = len(rows)

        name = DATASET_NAMES[split]
        entries_local[name] = dataset_entry(dst.name)  # relative to dataset_dir
        entries_official[name] = dataset_entry(str(dst))  # absolute path for optional mirror

    authoritative_info = dataset_dir / "dataset_info.json"
    update_registry(authoritative_info, entries_local)

    report: dict[str, Any] = {
        "mode": "b1_direct_only",
        "dataset_dir": str(dataset_dir),
        "authoritative_dataset_info": str(authoritative_info),
        "outputs": outputs,
        "dataset_names": DATASET_NAMES,
        "samples": counts,
        "socratic_required": False,
        "training_contract": (
            "The generated YAML must use dataset_dir equal to this dataset_dir; "
            "LLaMA-Factory reads dataset_info.json from that directory."
        ),
    }

    if args.register:
        lf_root = resolve_llama_factory_dir(settings, settings["training"]["base_model_key"])
        mirror_info = lf_root / "data" / "dataset_info.json"
        backup = update_registry(mirror_info, entries_official, ".re_scripts_v4_3_1_backup")
        report["official_registry_mirror"] = str(mirror_info)
        report["official_registry_backup"] = str(backup) if backup else None
        report["official_registry_note"] = "Optional mirror only; generated training YAML uses the external authoritative dataset_dir."

    report_path = dataset_dir / "dota128_direct_only_report.json"
    write_json(report_path, report)
    print("[DIRECT ONLY PREPARE] PASS")
    print(f"  train={counts['train']} -> {outputs['train']}")
    print(f"  val={counts['val']} -> {outputs['val']}")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  authoritative dataset_info: {authoritative_info}")
    print(f"  official mirror: {bool(args.register)}")
    print(f"  report: {report_path}")
    print("  next: python main_layer/run.py validate-lf-contract --settings settings.json --target b1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
