#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用官方 RS-EoT-7B Transformers 推理方式，批量评估 VRSBench-VQA。

特点：
1. 直接读取官方 VRSBench_EVAL_vqa.json，不需要转换 ShareGPT。
2. 每生成一条结果就追加到 JSONL，断电/中断后可续跑。
3. 保存 question_id、type、ground_truth、原始输出和 </think> 后最终答案。
4. 支持每题生成 K 次，用于 Avg@5 / Conv@5 / Pass@5。
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="本地 RS-EoT-7B 模型目录，或 ShaoRun/RS-EoT-7B")
    parser.add_argument("--json", required=True, help="VRSBench_EVAL_vqa.json")
    parser.add_argument("--image-dir", required=True, help="Images_val 目录")
    parser.add_argument("--output", required=True, help="预测结果 JSONL")
    parser.add_argument("--error-output", default=None, help="错误日志 JSONL；默认 output.errors.jsonl")
    parser.add_argument("--num-samples", type=int, default=1, help="每道题生成次数；论文三指标需要 5")
    parser.add_argument("--max-questions", type=int, default=None, help="只处理前 N 道题；调试时使用")
    parser.add_argument("--start-index", type=int, default=0, help="从 JSON 的第几个样本开始")
    parser.add_argument("--end-index", type=int, default=None, help="处理到 JSON 的第几个样本，不含该位置")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码；仅适合 Pass@1 调试")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def extract_final_answer(raw_text):
    text = (raw_text or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    # 常见答案标签兼容
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


def load_completed(output_path):
    completed = set()
    if not output_path.exists():
        return completed

    bad_lines = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    completed.add((int(row["question_id"]), int(row["sample_id"])))
            except Exception:
                bad_lines += 1
                print(f"[警告] 跳过无法解析的旧结果行：{line_no}", file=sys.stderr)

    print(f"[续跑] 已完成 {len(completed)} 个 (question_id, sample_id) 组合")
    if bad_lines:
        print(f"[续跑] 发现 {bad_lines} 行损坏记录；不会把它们视为完成")
    return completed


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU。请先运行 nvidia-smi 并检查 PyTorch CUDA。")

    if args.num_samples < 1:
        raise ValueError("--num-samples 必须 >= 1")

    model_path = args.model
    json_path = Path(args.json).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    error_path = (
        Path(args.error_output).expanduser().resolve()
        if args.error_output
        else output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    )

    if not json_path.is_file():
        raise FileNotFoundError(f"找不到 JSON：{json_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"找不到图片目录：{image_dir}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("VRSBench JSON 顶层必须是 list")

    start = max(0, args.start_index)
    end = len(data) if args.end_index is None else min(len(data), args.end_index)
    selected = data[start:end]
    if args.max_questions is not None:
        selected = selected[: args.max_questions]

    completed = load_completed(output_path)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("=" * 80)
    print("RS-EoT-7B × VRSBench-VQA 批量推理")
    print("=" * 80)
    print(f"GPU                 : {torch.cuda.get_device_name(0)}")
    print(f"dtype               : {dtype}")
    print(f"模型                : {model_path}")
    print(f"JSON                : {json_path}")
    print(f"图片目录            : {image_dir}")
    print(f"输出                : {output_path}")
    print(f"本次题目数          : {len(selected)}")
    print(f"每题生成次数        : {args.num_samples}")
    print(f"max_new_tokens      : {args.max_new_tokens}")
    print(f"greedy              : {args.greedy}")
    if not args.greedy:
        print(f"temperature/top_p/k : {args.temperature}/{args.top_p}/{args.top_k}")
    print("=" * 80)

    print("\n[1/2] 加载 processor ...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )

    print("[2/2] 加载 model ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    total_targets = len(selected) * args.num_samples
    pending = sum(
        1
        for item in selected
        for sample_id in range(args.num_samples)
        if (int(item["question_id"]), sample_id) not in completed
    )
    print(f"\n目标生成总数：{total_targets}；待生成：{pending}")

    progress = tqdm(total=pending, desc="推理", unit="条")
    start_time = time.time()

    for item in selected:
        qid = int(item["question_id"])
        image_id = str(item["image_id"])
        image_path = image_dir / image_id
        question = str(item["question"])
        gt = str(item["ground_truth"])
        qtype = str(item.get("type", ""))

        if not image_path.is_file():
            err = {
                "status": "error",
                "question_id": qid,
                "image_id": image_id,
                "error": f"图片不存在：{image_path}",
            }
            append_jsonl(error_path, err)
            print(f"\n[错误] {err['error']}", file=sys.stderr)
            continue

        for sample_id in range(args.num_samples):
            key = (qid, sample_id)
            if key in completed:
                continue

            one_seed = args.seed + qid * 1009 + sample_id
            set_seed(one_seed)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            try:
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                generate_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                }
                if args.greedy:
                    generate_kwargs.update(
                        {
                            "do_sample": False,
                        }
                    )
                else:
                    generate_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                        }
                    )

                t0 = time.time()
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **generate_kwargs)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                raw_output = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                final_answer = extract_final_answer(raw_output)

                elapsed = time.time() - t0
                new_tokens = int(generated_ids_trimmed[0].numel())

                row = {
                    "status": "ok",
                    "question_id": qid,
                    "sample_id": sample_id,
                    "seed": one_seed,
                    "image_id": image_id,
                    "type": qtype,
                    "dataset": item.get("dataset", "RSBench"),
                    "question": question,
                    "ground_truth": gt,
                    "raw_output": raw_output,
                    "final_answer": final_answer,
                    "new_tokens": new_tokens,
                    "elapsed_seconds": round(elapsed, 4),
                    "generation_config": {
                        "max_new_tokens": args.max_new_tokens,
                        "greedy": args.greedy,
                        "temperature": None if args.greedy else args.temperature,
                        "top_p": None if args.greedy else args.top_p,
                        "top_k": None if args.greedy else args.top_k,
                    },
                }
                append_jsonl(output_path, row)
                completed.add(key)

            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                err = {
                    "status": "error",
                    "question_id": qid,
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "error_type": "CUDAOutOfMemoryError",
                    "error": str(exc),
                }
                append_jsonl(error_path, err)
                print(
                    "\n[显存不足] 请降低 --max-new-tokens，或确认没有其他进程占显存。",
                    file=sys.stderr,
                )

            except Exception as exc:
                err = {
                    "status": "error",
                    "question_id": qid,
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, err)
                print(
                    f"\n[错误] qid={qid}, sample={sample_id}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            finally:
                for name in ["inputs", "generated_ids", "generated_ids_trimmed"]:
                    if name in locals():
                        del locals()[name]
                progress.update(1)

    progress.close()
    elapsed_all = time.time() - start_time
    print("\n" + "=" * 80)
    print("本次运行结束")
    print(f"输出文件：{output_path}")
    print(f"错误文件：{error_path}")
    print(f"本次耗时：{elapsed_all / 3600:.2f} 小时")
    print("重新执行同一条命令会自动跳过已完成项目。")
    print("=" * 80)


if __name__ == "__main__":
    main()
