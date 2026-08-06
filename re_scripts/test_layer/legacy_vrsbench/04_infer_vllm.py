#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 vLLM 批量推理 RS-EoT-7B × VRSBench-VQA。

适用场景：
- 10 条 Transformers 脚本验证通过后，进行全量 K=1 或 K=5。
- 单卡运行，按 batch 分批提交图文请求。
- JSONL 实时保存并支持续跑。

需要：
pip install vllm qwen-vl-utils transformers pillow tqdm
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--error-output", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def extract_final_answer(raw_text):
    text = (raw_text or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    lower = text.lower()
    for open_tag, close_tag in [
        ("<answer>", "</answer>"),
        ("<final>", "</final>"),
    ]:
        if open_tag in lower:
            start = lower.rfind(open_tag) + len(open_tag)
            end = lower.find(close_tag, start)
            if end != -1:
                return text[start:end].strip()
    return text


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed(path):
    completed = defaultdict(set)
    if not path.exists():
        return completed

    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    completed[int(row["question_id"])].add(int(row["sample_id"]))
            except Exception:
                bad += 1
                print(f"[警告] 旧结果第 {line_no} 行损坏，已跳过", file=sys.stderr)

    done_pairs = sum(len(v) for v in completed.values())
    print(f"[续跑] 已有 {done_pairs} 个完成的 (question_id, sample_id)")
    if bad:
        print(f"[续跑] 损坏行数：{bad}")
    return completed


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples 必须 >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须 >= 1")

    json_path = Path(args.json).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    error_path = (
        Path(args.error_output).expanduser().resolve()
        if args.error_output
        else output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    )

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    start = max(0, args.start_index)
    end = len(data) if args.end_index is None else min(len(data), args.end_index)
    selected = data[start:end]
    if args.max_questions is not None:
        selected = selected[:args.max_questions]

    completed = load_completed(output_path)
    pending_items = [
        item for item in selected
        if len(completed[int(item["question_id"])]) < args.num_samples
    ]

    print("=" * 80)
    print("vLLM：RS-EoT-7B × VRSBench-VQA")
    print("=" * 80)
    print(f"模型                 : {args.model}")
    print(f"题目总数             : {len(selected)}")
    print(f"尚未完整完成的题目   : {len(pending_items)}")
    print(f"每题生成             : {args.num_samples}")
    print(f"batch size           : {args.batch_size}")
    print(f"max tokens           : {args.max_tokens}")
    print(f"max model len        : {args.max_model_len}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print("=" * 80)

    print("[1/2] 加载 processor")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print("[2/2] 启动 vLLM")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    total_batches = math.ceil(len(pending_items) / args.batch_size) if pending_items else 0
    pbar = tqdm(total=len(pending_items), desc="题目", unit="题")
    started = time.time()

    for batch_no, batch in enumerate(chunks(pending_items, args.batch_size), 1):
        llm_inputs = []
        valid_items = []

        for item in batch:
            qid = int(item["question_id"])
            image_path = image_dir / str(item["image_id"])
            if not image_path.is_file():
                append_jsonl(error_path, {
                    "status": "error",
                    "question_id": qid,
                    "image_id": item["image_id"],
                    "error": f"图片不存在：{image_path}",
                })
                pbar.update(1)
                continue

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": str(item["question"])},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)

                mm_data = {}
                if image_inputs is not None:
                    if isinstance(image_inputs, list) and len(image_inputs) == 1:
                        mm_data["image"] = image_inputs[0]
                    else:
                        mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs

                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                })
                valid_items.append(item)

            except Exception as exc:
                append_jsonl(error_path, {
                    "status": "error",
                    "question_id": qid,
                    "image_id": item["image_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                pbar.update(1)

        if not llm_inputs:
            continue

        try:
            batch_started = time.time()
            request_outputs = llm.generate(
                llm_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            batch_elapsed = time.time() - batch_started

            for item, req_out in zip(valid_items, request_outputs):
                qid = int(item["question_id"])
                already = completed[qid]

                for sample_id, candidate in enumerate(req_out.outputs):
                    if sample_id in already:
                        continue

                    raw_output = candidate.text
                    row = {
                        "status": "ok",
                        "question_id": qid,
                        "sample_id": sample_id,
                        "seed": args.seed,
                        "image_id": str(item["image_id"]),
                        "type": str(item.get("type", "")),
                        "dataset": item.get("dataset", "RSBench"),
                        "question": str(item["question"]),
                        "ground_truth": str(item["ground_truth"]),
                        "raw_output": raw_output,
                        "final_answer": extract_final_answer(raw_output),
                        "finish_reason": getattr(candidate, "finish_reason", None),
                        "new_tokens": len(getattr(candidate, "token_ids", []) or []),
                        "batch_elapsed_seconds": round(batch_elapsed, 4),
                        "generation_config": {
                            "n": args.num_samples,
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "max_model_len": args.max_model_len,
                            "batch_size": args.batch_size,
                        },
                    }
                    append_jsonl(output_path, row)
                    already.add(sample_id)

                pbar.update(1)

        except Exception as exc:
            for item in valid_items:
                append_jsonl(error_path, {
                    "status": "error",
                    "question_id": int(item["question_id"]),
                    "image_id": str(item["image_id"]),
                    "batch_no": batch_no,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                pbar.update(1)

            print(
                f"\n[批次失败] batch={batch_no}/{total_batches}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print(
                "[建议] 如为显存不足，将 --batch-size 改为 4、2 或 1，"
                "并重新运行同一命令续跑。",
                file=sys.stderr,
            )

        finally:
            del llm_inputs
            gc.collect()

    pbar.close()
    print("\n" + "=" * 80)
    print("vLLM 推理结束")
    print(f"输出：{output_path}")
    print(f"错误：{error_path}")
    print(f"耗时：{(time.time() - started) / 3600:.2f} 小时")
    print("再次执行相同命令可续跑。")
    print("=" * 80)


if __name__ == "__main__":
    main()
