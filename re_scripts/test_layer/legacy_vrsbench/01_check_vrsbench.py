#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 VRSBench-VQA 测试集是否完整。

预期：
- JSON 问题数：37409
- 唯一 question_id：37409，范围 0~37408
- JSON 引用的唯一图片：9349
- 磁盘图片：9350
- JSON 引用但磁盘缺失：0
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys


REQUIRED_FIELDS = {
    "image_id",
    "question",
    "ground_truth",
    "dataset",
    "question_id",
    "type",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="VRSBench_EVAL_vqa.json 路径")
    parser.add_argument("--image-dir", required=True, help="Images_val 图片目录")
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()

    print("=" * 72)
    print("VRSBench-VQA 数据完整性检查")
    print("=" * 72)
    print(f"JSON:      {json_path}")
    print(f"图片目录:  {image_dir}")

    if not json_path.is_file():
        print(f"\n[失败] JSON 文件不存在：{json_path}", file=sys.stderr)
        sys.exit(1)
    if not image_dir.is_dir():
        print(f"\n[失败] 图片目录不存在：{image_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"\n[失败] JSON 无法解析：{exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(f"\n[失败] JSON 顶层应当是 list，实际是 {type(data).__name__}", file=sys.stderr)
        sys.exit(1)

    qids = []
    json_images = []
    missing_fields = []
    blank_fields = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            missing_fields.append((idx, "整条数据不是对象"))
            continue

        missing = sorted(REQUIRED_FIELDS - set(item))
        if missing:
            missing_fields.append((idx, ",".join(missing)))

        for key in REQUIRED_FIELDS:
            if key in item and item[key] in (None, ""):
                blank_fields.append((idx, key))

        if "question_id" in item:
            qids.append(item["question_id"])
        if "image_id" in item:
            json_images.append(str(item["image_id"]))

    qid_counts = Counter(qids)
    duplicate_qids = sorted([qid for qid, count in qid_counts.items() if count > 1])

    integer_qids = [qid for qid in qids if isinstance(qid, int)]
    min_qid = min(integer_qids) if integer_qids else None
    max_qid = max(integer_qids) if integer_qids else None

    expected_qids = set(range(min_qid, max_qid + 1)) if integer_qids else set()
    missing_qids = sorted(expected_qids - set(integer_qids))

    disk_paths = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    disk_names = [p.name for p in disk_paths]

    disk_name_counts = Counter(disk_names)
    duplicate_disk_names = sorted(
        [name for name, count in disk_name_counts.items() if count > 1]
    )

    json_image_set = set(json_images)
    disk_name_set = set(disk_names)
    missing_images = sorted(json_image_set - disk_name_set)
    unused_images = sorted(disk_name_set - json_image_set)

    print("\n[统计]")
    print(f"JSON 问题总数                 : {len(data)}")
    print(f"question_id 唯一数量          : {len(set(qids))}")
    print(f"question_id 最小值/最大值     : {min_qid} / {max_qid}")
    print(f"重复 question_id 数量         : {len(duplicate_qids)}")
    print(f"区间内缺失 question_id 数量   : {len(missing_qids)}")
    print(f"JSON 引用的唯一图片数量       : {len(json_image_set)}")
    print(f"磁盘图片文件数量              : {len(disk_paths)}")
    print(f"磁盘唯一图片文件名数量        : {len(disk_name_set)}")
    print(f"JSON 引用但磁盘缺失的图片数量 : {len(missing_images)}")
    print(f"磁盘存在但 JSON 未引用数量    : {len(unused_images)}")
    print(f"缺失字段的样本数量            : {len(missing_fields)}")
    print(f"空字段数量                    : {len(blank_fields)}")
    print(f"磁盘重复文件名数量            : {len(duplicate_disk_names)}")

    if missing_images:
        print("\n[前 20 个缺失图片]")
        for name in missing_images[:20]:
            print(name)

    if unused_images:
        print("\n[磁盘存在但 VQA JSON 未使用的图片]")
        for name in unused_images[:20]:
            print(name)

    if duplicate_qids:
        print("\n[前 20 个重复 question_id]")
        print(duplicate_qids[:20])

    if missing_qids:
        print("\n[前 20 个缺失 question_id]")
        print(missing_qids[:20])

    if missing_fields:
        print("\n[前 20 个缺失字段样本]")
        for row in missing_fields[:20]:
            print(row)

    ok = (
        len(data) == 37409
        and len(set(qids)) == 37409
        and min_qid == 0
        and max_qid == 37408
        and not duplicate_qids
        and not missing_qids
        and len(json_image_set) == 9349
        and len(disk_paths) == 9350
        and not missing_images
        and not missing_fields
        and not blank_fields
    )

    print("\n" + "=" * 72)
    if ok:
        print("[通过] 数据结构与当前 VRSBench-VQA 测试集预期一致。")
        sys.exit(0)
    else:
        print("[未完全通过] 请根据上面的统计检查路径、解压结果或 JSON 文件。")
        sys.exit(2)


if __name__ == "__main__":
    main()
