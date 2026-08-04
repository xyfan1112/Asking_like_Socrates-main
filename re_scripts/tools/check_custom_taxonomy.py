#!/usr/bin/env python3
"""Validate classes.txt and OBB class ids before running the data pipeline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main_layer"))

from common import CLASSES_FILE, DOTA_CLASSES, QA_LANGUAGE, TAXONOMY_SHA256, load_settings, settings_from_cli


def _label_dirs(dataset_root: Path, split: str) -> list[Path]:
    candidates = [
        dataset_root / split / "labels",
        dataset_root / "labels" / split,
    ]
    return [path for path in candidates if path.is_dir()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--max-examples", type=int, default=20)
    args = ap.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    dataset_root = Path(settings["paths"]["dota128_root"]).expanduser().resolve()
    splits = settings.get("data_conversion", {}).get("splits", ["train", "val"])

    print("[CUSTOM TAXONOMY CHECK]")
    print(f"  settings_dataset_root: {dataset_root}")
    print(f"  classes_file: {CLASSES_FILE or '(built-in DOTA fallback)'}")
    print(f"  qa_language: {QA_LANGUAGE}")
    print(f"  num_classes: {len(DOTA_CLASSES)}")
    print(f"  taxonomy_sha256: {TAXONOMY_SHA256}")
    print("  class_id -> canonical label:")
    for class_id, label in enumerate(DOTA_CLASSES):
        print(f"    {class_id}: {label}")

    errors: list[str] = []
    warnings: list[str] = []
    id_counts: Counter[int] = Counter()
    examples: defaultdict[int, list[str]] = defaultdict(list)
    total_files = 0
    total_objects = 0

    if not dataset_root.is_dir():
        errors.append(f"dataset root 不存在: {dataset_root}")

    for split in splits:
        dirs = _label_dirs(dataset_root, str(split))
        if not dirs:
            errors.append(
                f"{split}: 找不到标签目录；检查过 "
                f"{dataset_root / str(split) / 'labels'} 和 "
                f"{dataset_root / 'labels' / str(split)}"
            )
            continue
        for label_dir in dirs:
            paths = sorted(label_dir.glob("*.txt"))
            if not paths:
                warnings.append(f"{split}: 标签目录为空: {label_dir}")
            for path in paths:
                total_files += 1
                try:
                    lines = path.read_text(encoding="utf-8-sig").splitlines()
                except UnicodeDecodeError as exc:
                    errors.append(f"{split}:{path}: 非UTF-8标签文件: {exc}")
                    continue
                for line_no, raw in enumerate(lines, 1):
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 9:
                        errors.append(
                            f"{split}:{path.name}:{line_no}: 需要9列 "
                            "class_id x1 y1 x2 y2 x3 y3 x4 y4，"
                            f"实际{len(parts)}列: {line[:160]}"
                        )
                        continue
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        errors.append(
                            f"{split}:{path.name}:{line_no}: class_id不是整数: {parts[0]!r}"
                        )
                        continue
                    try:
                        coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        errors.append(
                            f"{split}:{path.name}:{line_no}: OBB坐标不是数值: {line[:160]}"
                        )
                        continue
                    total_objects += 1
                    id_counts[class_id] += 1
                    if len(examples[class_id]) < 3:
                        examples[class_id].append(f"{split}/{path.name}:{line_no}")
                    if class_id < 0 or class_id >= len(DOTA_CLASSES):
                        errors.append(
                            f"{split}:{path.name}:{line_no}: class_id={class_id} 越界；"
                            f"当前classes.txt仅允许0..{len(DOTA_CLASSES)-1}"
                        )
                    if any(not (-1e12 < value < 1e12) for value in coords):
                        errors.append(
                            f"{split}:{path.name}:{line_no}: 坐标包含非有限或异常数值"
                        )

    print(f"  label_files: {total_files}")
    print(f"  parsed_objects: {total_objects}")
    if id_counts:
        print(f"  used_id_range: {min(id_counts)}..{max(id_counts)}")
        print("  used class ids:")
        for class_id in sorted(id_counts):
            status = (
                DOTA_CLASSES[class_id]
                if 0 <= class_id < len(DOTA_CLASSES)
                else "<OUT_OF_RANGE>"
            )
            print(
                f"    id={class_id} count={id_counts[class_id]} "
                f"label={status!r} examples={examples[class_id]}"
            )

    unused = [
        (idx, label) for idx, label in enumerate(DOTA_CLASSES)
        if id_counts.get(idx, 0) == 0
    ]
    if unused:
        warnings.append(
            "classes.txt中存在当前train/val未使用的类别: "
            + ", ".join(f"{idx}:{label}" for idx, label in unused[:50])
        )

    if warnings:
        print("  warnings:")
        for item in warnings:
            print(f"    - {item}")
    if errors:
        print("  errors:")
        for item in errors[: args.max_examples]:
            print(f"    - {item}")
        if len(errors) > args.max_examples:
            print(f"    ... 其余 {len(errors)-args.max_examples} 个错误省略")
        print("[RESULT] FAIL")
        return 2

    print("[RESULT] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
