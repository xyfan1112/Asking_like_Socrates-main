#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对预测 JSONL 做本地、可复现的规则匹配评估。

说明：
- 这是无需外部 API 的本地评估器。
- 它能算 Avg@K / Conv@K / Pass@K，并按 type 分组。
- VRSBench 官方 evaluator 对部分自由文本答案会使用语义判定；
  因此本脚本结果不应冒充论文的完全同协议结果。
"""

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


ARTICLES = {"a", "an", "the"}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="02_infer_vrsbench.py 生成的 JSONL")
    parser.add_argument("--k", type=int, default=1, help="每题使用前 K 个 sample")
    parser.add_argument("--save-summary", default=None, help="保存汇总 JSON")
    return parser.parse_args()


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"</?[^>]+>", " ", text)

    # 去掉常见答案前缀
    text = re.sub(
        r"^(final\s+answer|the\s+answer\s+is|answer)\s*[:：]?\s*",
        "",
        text,
    )

    # 标点转空格
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for tok in text.split():
        if tok in ARTICLES:
            continue
        tok = NUMBER_WORDS.get(tok, tok)
        tokens.append(tok)
    return " ".join(tokens)


def token_phrase_in(needle, haystack):
    if not needle or not haystack:
        return False
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def first_yes_no(text):
    match = re.search(r"(?<!\w)(yes|no)(?!\w)", text)
    return match.group(1) if match else None


def first_number(text):
    match = re.search(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", text)
    return match.group(0) if match else None


def is_correct(ground_truth, prediction):
    gt = normalize(ground_truth)
    pred = normalize(prediction)

    if not gt or not pred:
        return False

    if gt == pred:
        return True

    if gt in {"yes", "no"}:
        return first_yes_no(pred) == gt

    if re.fullmatch(r"-?\d+(?:\.\d+)?", gt):
        return first_number(pred) == gt

    # 对自由文本允许完整标准答案作为独立短语出现在预测中
    return token_phrase_in(gt, pred)


def load_rows(path):
    rows = []
    bad = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    rows.append(row)
            except Exception:
                bad += 1
                print(f"[警告] 第 {line_no} 行无法解析")
    return rows, bad


def metrics_for_groups(groups, k):
    usable = {}
    incomplete = 0

    for qid, rows in groups.items():
        by_sid = {}
        for row in rows:
            sid = int(row["sample_id"])
            by_sid[sid] = row

        chosen = [by_sid[sid] for sid in range(k) if sid in by_sid]
        if len(chosen) != k:
            incomplete += 1
            continue

        scored = []
        for row in chosen:
            correct = is_correct(row["ground_truth"], row["final_answer"])
            scored.append((row, correct))
        usable[qid] = scored

    if not usable:
        return None, incomplete

    all_correct = [
        correct
        for scored in usable.values()
        for _, correct in scored
    ]

    per_question_counts = [
        sum(correct for _, correct in scored)
        for scored in usable.values()
    ]

    avg_at_k = sum(all_correct) / len(all_correct)
    majority_threshold = math.floor(k / 2) + 1
    conv_at_k = (
        sum(count >= majority_threshold for count in per_question_counts)
        / len(per_question_counts)
    )
    pass_at_k = (
        sum(count >= 1 for count in per_question_counts)
        / len(per_question_counts)
    )

    return {
        "questions": len(usable),
        "generations": len(all_correct),
        f"Avg@{k}": avg_at_k,
        f"Conv@{k}": conv_at_k,
        f"Pass@{k}": pass_at_k,
        "majority_threshold": majority_threshold,
    }, incomplete


def main():
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k 必须 >= 1")

    rows, bad = load_rows(args.pred)
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["question_id"])].append(row)

    overall, incomplete = metrics_for_groups(groups, args.k)
    if overall is None:
        raise RuntimeError(f"没有找到每题至少 {args.k} 条完整预测的数据")

    type_groups = defaultdict(lambda: defaultdict(list))
    for qid, qrows in groups.items():
        qtype = str(qrows[0].get("type", "unknown"))
        type_groups[qtype][qid].extend(qrows)

    by_type = {}
    for qtype in sorted(type_groups):
        result, _ = metrics_for_groups(type_groups[qtype], args.k)
        if result is not None:
            by_type[qtype] = result

    print("=" * 80)
    print("VRSBench-VQA 本地规则评估")
    print("=" * 80)
    print(f"预测文件             : {Path(args.pred).resolve()}")
    print(f"K                    : {args.k}")
    print(f"可用完整题目数       : {overall['questions']}")
    print(f"不完整、暂未计分题目 : {incomplete}")
    print(f"损坏 JSONL 行        : {bad}")
    print(f"Avg@{args.k}               : {overall[f'Avg@{args.k}'] * 100:.2f}")
    print(f"Conv@{args.k}              : {overall[f'Conv@{args.k}'] * 100:.2f}")
    print(f"Pass@{args.k}              : {overall[f'Pass@{args.k}'] * 100:.2f}")

    print("\n[按问题类型]")
    print(f"{'type':30s} {'N':>7s} {'Avg':>8s} {'Conv':>8s} {'Pass':>8s}")
    for qtype, result in by_type.items():
        print(
            f"{qtype[:30]:30s} "
            f"{result['questions']:7d} "
            f"{result[f'Avg@{args.k}'] * 100:8.2f} "
            f"{result[f'Conv@{args.k}'] * 100:8.2f} "
            f"{result[f'Pass@{args.k}'] * 100:8.2f}"
        )

    summary = {
        "protocol": "local_rule_based_not_official_semantic_judge",
        "k": args.k,
        "overall": overall,
        "incomplete_questions": incomplete,
        "bad_jsonl_lines": bad,
        "by_type": by_type,
    }

    if args.save_summary:
        path = Path(args.save_summary).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n汇总已保存：{path}")

    print("\n注意：这是无 API 的本地规则分数。")
    print("VRSBench 官方自由文本评估可能对同义表达做语义判定，论文分数不能仅凭此脚本完全复现。")


if __name__ == "__main__":
    main()
