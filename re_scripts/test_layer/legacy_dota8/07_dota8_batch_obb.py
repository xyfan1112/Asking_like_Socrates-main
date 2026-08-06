#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DOTA8：Qwen2.5-VL-7B-Instruct 与 RS-EoT-7B 的批量 OBB 推理。

固定实验：
- 8 张 DOTA8 图片（train 4 + val 4）
- 2 个模型：Qwen / RS
- 2 个提示模式：direct / prompt
- 总推理次数：8 × 2 × 2 = 32

运行时不需要输入任何参数。只需修改下方“配置区”的路径。
"""

import csv
import gc
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ============================================================
# 配置区：路径不同只修改这里
# ============================================================

DATASET_ROOT = Path("/home/vieo/vieo/fxy_workspace/fxy/datasets/dota8")
OUTPUT_ROOT = Path("/home/vieo/vieo/fxy_workspace/fxy/dota8_vlm_obb_results")

MODELS = {
    "Qwen": Path("/home/vieo/vieo/fxy_workspace/fxy/models/Qwen2.5-VL-7B-Instruct"),
    "RS": Path("/home/vieo/vieo/fxy_workspace/fxy/models/RS-EoT-7B"),
}

MODES = ("direct", "prompt")

DEVICE = "cuda:0"

# DOTA8 中 P0861 有 100 余个目标，最终输出会较长。
MAX_NEW_TOKENS = 8192

# 保留接近原始 1024×1024 的视觉分辨率。
MIN_VISUAL_TOKENS = 256
MAX_VISUAL_TOKENS = 1536

# auto / fp16 / bf16
DTYPE = "auto"

# 已有有效 JSON 和可视化图片时自动跳过。
SKIP_EXISTING = True

# 空闲显存低于该值就停止，避免 CPU offload。
MIN_FREE_GPU_GB = 20.0

# 终端默认不打印完整 think，完整输出会写入 JSON。
SHOW_RAW_IN_TERMINAL = False

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp",
}

CLASS_NAMES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

CLASS_ALIASES = {
    "storage-tank": "storage tank",
    "baseball-diamond": "baseball diamond",
    "tennis-court": "tennis court",
    "basketball-court": "basketball court",
    "ground-track-field": "ground track field",
    "large-vehicle": "large vehicle",
    "small-vehicle": "small vehicle",
    "soccer-ball-field": "soccer ball field",
    "swimming-pool": "swimming pool",
}


# ============================================================
# 提示词
# ============================================================

def build_question(mode: str) -> str:
    class_text = ", ".join(CLASS_NAMES)

    common = f"""
Detect every object belonging to the following DOTA classes:
{class_text}

For every detected object, output:
class_name|confidence|x1,y1,x2,y2,x3,y3,x4,y4

Requirements:
1. Coordinates must be pixel coordinates in the original image.
2. The four points must follow the object's oriented boundary in clockwise or counter-clockwise order.
3. confidence must be a number between 0 and 1.
4. Use exactly the class names listed above.
5. Do not merge nearby objects.
6. The final answer must be enclosed by these markers:

FINAL_DETECTIONS
class_name|confidence|x1,y1,x2,y2,x3,y3,x4,y4
END_DETECTIONS

If there is no object, output:
FINAL_DETECTIONS
END_DETECTIONS
""".strip()

    if mode == "direct":
        return common

    if mode == "prompt":
        return (
            "The aerial image may contain small, densely arranged, "
            "partially occluded, edge-truncated, or visually similar objects. "
            "First scan the whole image, then inspect local regions carefully, "
            "and finally verify that no valid object has been missed or counted twice.\n\n"
            + common
        )

    raise ValueError(f"未知 mode：{mode}")


# ============================================================
# DOTA8 文件与标签
# ============================================================

def collect_samples() -> List[Dict]:
    samples = []

    for split in ("train", "val"):
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        if not image_dir.is_dir():
            raise NotADirectoryError(f"图片目录不存在：{image_dir}")
        if not label_dir.is_dir():
            raise NotADirectoryError(f"标签目录不存在：{label_dir}")

        for image_path in sorted(image_dir.iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.is_file():
                raise FileNotFoundError(
                    f"图片缺少同名标签：{image_path} -> {label_path}"
                )

            samples.append(
                {
                    "split": split,
                    "image_path": image_path.resolve(),
                    "label_path": label_path.resolve(),
                    "image_key": f"{split}/{image_path.name}",
                }
            )

    return samples


def load_ground_truth(
    label_path: Path,
    width: int,
    height: int,
) -> List[Dict]:
    objects = []

    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) != 9:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行不是 "
                    "class + 8 坐标的 YOLO OBB 格式"
                )

            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]

            points = []

            for index in range(0, 8, 2):
                x = values[index]
                y = values[index + 1]

                # DOTA8 标签是归一化坐标。
                x *= width
                y *= height

                points.append([x, y])

            class_name = (
                CLASS_NAMES[class_id]
                if 0 <= class_id < len(CLASS_NAMES)
                else f"class_{class_id}"
            )

            objects.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "points": points,
                }
            )

    return objects


# ============================================================
# 模型输出解析
# ============================================================

def extract_final_answer(raw_text: str) -> str:
    text = str(raw_text or "").strip()

    # RS-EoT 通常把最终答案放在 </think> 后。
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    lower = text.lower()

    for open_tag, close_tag in (
        ("<answer>", "</answer>"),
        ("<final>", "</final>"),
    ):
        if open_tag in lower:
            start = lower.rfind(open_tag) + len(open_tag)
            end = lower.find(close_tag, start)

            if end != -1:
                return text[start:end].strip()

    return text


def normalize_class_name(text: str) -> Optional[str]:
    name = str(text).strip().lower()
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)

    if name in CLASS_ALIASES:
        name = CLASS_ALIASES[name]

    # 去掉末尾常见标点。
    name = name.strip(" .,:;[](){}\"'")

    if name in CLASS_TO_ID:
        return name

    # 尝试把连字符改为空格。
    name2 = name.replace("-", " ")
    name2 = re.sub(r"\s+", " ", name2).strip()

    if name2 in CLASS_TO_ID:
        return name2

    return None


def parse_detection_line(line: str) -> Optional[Dict]:
    """
    解析：
    class_name|confidence|x1,y1,x2,y2,x3,y3,x4,y4

    同时兼容模型偶尔只输出 4 个 xyxy 数字；
    此时会转成水平矩形 OBB，并记录 xyxy_fallback。
    """

    text = line.strip().strip("`")

    if not text or "|" not in text:
        return None

    parts = [part.strip() for part in text.split("|")]

    if len(parts) < 3:
        return None

    class_name = normalize_class_name(parts[0])

    if class_name is None:
        return None

    try:
        confidence = float(
            re.search(
                r"-?\d+(?:\.\d+)?",
                parts[1],
            ).group(0)
        )
    except Exception:
        confidence = 0.5

    confidence = max(0.0, min(1.0, confidence))

    number_text = "|".join(parts[2:])

    numbers = [
        float(value)
        for value in re.findall(
            r"-?\d+(?:\.\d+)?",
            number_text,
        )
    ]

    box_format = "obb8"

    if len(numbers) >= 8:
        numbers = numbers[:8]
        points = [
            [numbers[0], numbers[1]],
            [numbers[2], numbers[3]],
            [numbers[4], numbers[5]],
            [numbers[6], numbers[7]],
        ]

    elif len(numbers) == 4:
        # 模型若退化成水平框，转换为 4 点 OBB。
        x1, y1, x2, y2 = numbers
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])

        points = [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]
        box_format = "xyxy_fallback"

    else:
        return None

    return {
        "class_id": CLASS_TO_ID[class_name],
        "class_name": class_name,
        "confidence": confidence,
        "points": points,
        "box_format": box_format,
        "source_line": line.strip(),
    }


def parse_detections(
    final_answer: str,
    raw_output: str,
) -> Tuple[List[Dict], str]:
    """
    优先解析 FINAL_DETECTIONS 与 END_DETECTIONS 之间的内容。
    """

    candidate = final_answer

    match = re.search(
        r"FINAL_DETECTIONS\s*(.*?)\s*END_DETECTIONS",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )

    source = "final_markers"

    if match:
        block = match.group(1)
    else:
        match = re.search(
            r"FINAL_DETECTIONS\s*(.*)",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            block = match.group(1)
            source = "final_marker_without_end"
        else:
            block = candidate
            source = "final_answer_fallback"

    detections = []

    for line in block.splitlines():
        parsed = parse_detection_line(line)

        if parsed is not None:
            detections.append(parsed)

    # 最终答案没有解析到时，才扫描完整输出。
    if not detections and raw_output != candidate:
        source = "raw_output_fallback"

        for line in raw_output.splitlines():
            parsed = parse_detection_line(line)

            if parsed is not None:
                detections.append(parsed)

    # 去掉完全重复的输出行。
    unique = []
    seen = set()

    for det in detections:
        key = (
            det["class_id"],
            round(det["confidence"], 6),
            tuple(
                round(value, 3)
                for point in det["points"]
                for value in point
            ),
        )

        if key not in seen:
            unique.append(det)
            seen.add(key)

    return unique, source


def sanitize_detections(
    detections: List[Dict],
    width: int,
    height: int,
) -> List[Dict]:
    """
    预测框坐标限制到原图范围。
    """

    cleaned = []

    for det in detections:
        points = []

        for x, y in det["points"]:
            x = max(0.0, min(float(width - 1), float(x)))
            y = max(0.0, min(float(height - 1), float(y)))
            points.append([x, y])

        # 至少存在非零宽高。
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        if max(xs) - min(xs) < 1.0:
            continue
        if max(ys) - min(ys) < 1.0:
            continue

        item = dict(det)
        item["points"] = points
        cleaned.append(item)

    return cleaned


# ============================================================
# 可视化
# ============================================================

def draw_polygon(
    draw: ImageDraw.ImageDraw,
    points: List[List[float]],
    color: Tuple[int, int, int],
    width: int,
):
    xy = [
        (int(round(x)), int(round(y)))
        for x, y in points
    ]

    if len(xy) != 4:
        return

    xy.append(xy[0])

    draw.line(
        xy,
        fill=color,
        width=width,
        joint="curve",
    )


def make_overlay(
    image: Image.Image,
    ground_truth: List[Dict],
    predictions: List[Dict],
    model_name: str,
    mode: str,
) -> Image.Image:
    """
    绿色：GT
    红色：预测
    """

    output = image.copy()
    draw = ImageDraw.Draw(output)

    width, height = image.size
    line_width = max(2, int(round(min(width, height) / 220)))

    # GT 绿色。
    for index, obj in enumerate(ground_truth, 1):
        draw_polygon(
            draw,
            obj["points"],
            color=(0, 255, 0),
            width=line_width,
        )

        x, y = obj["points"][0]

        draw.text(
            (int(x), max(0, int(y) - 13)),
            f"GT:{obj['class_name']}",
            fill=(0, 180, 0),
        )

    # 预测红色。
    for index, obj in enumerate(predictions, 1):
        draw_polygon(
            draw,
            obj["points"],
            color=(255, 0, 0),
            width=line_width,
        )

        x, y = obj["points"][0]

        draw.text(
            (int(x), max(0, int(y) + 2)),
            (
                f"P:{obj['class_name']} "
                f"{obj['confidence']:.2f}"
            ),
            fill=(255, 0, 0),
        )

    draw.rectangle(
        [0, 0, min(width - 1, 530), 22],
        fill=(255, 255, 255),
    )

    draw.text(
        (5, 4),
        (
            f"{model_name} | {mode} | "
            f"GT={len(ground_truth)} | "
            f"Pred={len(predictions)} | "
            "GT:green Pred:red"
        ),
        fill=(0, 0, 0),
    )

    return output


# ============================================================
# 模型加载与显存释放
# ============================================================

def choose_dtype():
    if DTYPE == "fp16":
        return torch.float16

    if DTYPE == "bf16":
        return torch.bfloat16

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def check_gpu_before_loading(model_name: str):
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU")

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3

    print(
        f"\n[{model_name}] GPU={torch.cuda.get_device_name(0)}，"
        f"空闲显存={free_gb:.2f}/{total_gb:.2f} GB"
    )

    if free_gb < MIN_FREE_GPU_GB:
        raise RuntimeError(
            f"当前空闲显存只有 {free_gb:.2f} GB，"
            f"低于要求的 {MIN_FREE_GPU_GB:.2f} GB。"
            "请先暂停或结束其他 GPU 任务。"
        )


def load_model(
    model_name: str,
    model_path: Path,
):
    check_gpu_before_loading(model_name)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path}"
        )

    dtype = choose_dtype()

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=MIN_VISUAL_TOKENS * 28 * 28,
        max_pixels=MAX_VISUAL_TOKENS * 28 * 28,
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    )

    model = model.to(DEVICE)
    model.eval()

    non_cuda = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device.type != "cuda"
    ]

    if non_cuda:
        raise RuntimeError(
            f"{model_name} 仍有 {len(non_cuda)} 个参数不在 GPU，"
            "已停止，避免 CPU offload。"
        )

    print(
        f"[{model_name}] 模型已加载，全部参数位于 GPU，dtype={dtype}"
    )

    return model, processor


def release_model(
    model,
    processor,
    model_name: str,
):
    print(f"\n[{model_name}] 释放模型")

    del model
    del processor

    gc.collect()
    torch.cuda.empty_cache()

    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    time.sleep(2)

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    print(
        f"[{model_name}] 释放后空闲显存："
        f"{free_bytes / 1024**3:.2f}/"
        f"{total_bytes / 1024**3:.2f} GB"
    )


# ============================================================
# 输出路径与续跑
# ============================================================

def result_paths(
    sample: Dict,
    model_name: str,
    mode: str,
):
    output_dir = (
        OUTPUT_ROOT
        / model_name
        / mode
        / sample["split"]
    )

    stem = sample["image_path"].stem

    return {
        "dir": output_dir,
        "json": output_dir / f"{stem}_result.json",
        "overlay": output_dir / f"{stem}_overlay.png",
        "error": output_dir / f"{stem}_error.json",
    }


def is_finished(
    sample: Dict,
    model_name: str,
    mode: str,
) -> bool:
    paths = result_paths(sample, model_name, mode)

    if (
        not paths["json"].is_file()
        or not paths["overlay"].is_file()
    ):
        return False

    try:
        with paths["json"].open("r", encoding="utf-8") as f:
            row = json.load(f)

        return row.get("status") == "ok"

    except Exception:
        return False


# ============================================================
# 单次推理
# ============================================================

def run_one(
    model,
    processor,
    model_name: str,
    model_path: Path,
    mode: str,
    sample: Dict,
) -> Dict:
    paths = result_paths(sample, model_name, mode)

    paths["dir"].mkdir(parents=True, exist_ok=True)

    image_path = sample["image_path"]
    label_path = sample["label_path"]

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    ground_truth = load_ground_truth(
        label_path,
        width,
        height,
    )

    question = build_question(mode)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(DEVICE)

    torch.cuda.synchronize()
    started = time.time()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - started

    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    raw_output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    final_answer = extract_final_answer(raw_output)

    predictions, parse_source = parse_detections(
        final_answer,
        raw_output,
    )

    predictions = sanitize_detections(
        predictions,
        width,
        height,
    )

    overlay = make_overlay(
        image,
        ground_truth,
        predictions,
        model_name,
        mode,
    )

    overlay.save(paths["overlay"])

    new_tokens = int(trimmed[0].numel())

    result = {
        "status": "ok",
        "model_name": model_name,
        "model_path": str(model_path),
        "mode": mode,
        "split": sample["split"],
        "image_key": sample["image_key"],
        "image_name": image_path.name,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "image_width": width,
        "image_height": height,
        "question": question,
        "raw_output": raw_output,
        "final_answer": final_answer,
        "parse_source": parse_source,
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predictions),
        "predictions": predictions,
        "elapsed_seconds": round(elapsed, 4),
        "new_tokens": new_tokens,
        "tokens_per_second": round(
            new_tokens / max(elapsed, 1e-8),
            4,
        ),
        "reached_max_new_tokens": (
            new_tokens >= MAX_NEW_TOKENS
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_visual_tokens": MAX_VISUAL_TOKENS,
        "overlay_image": str(paths["overlay"]),
        "result_json": str(paths["json"]),
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    with paths["json"].open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    if SHOW_RAW_IN_TERMINAL:
        tqdm.write(raw_output)

    del inputs
    del generated_ids
    del trimmed

    return result


def save_error(
    sample: Dict,
    model_name: str,
    model_path: Path,
    mode: str,
    exc: Exception,
):
    paths = result_paths(sample, model_name, mode)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    row = {
        "status": "error",
        "model_name": model_name,
        "model_path": str(model_path),
        "mode": mode,
        "split": sample["split"],
        "image_key": sample["image_key"],
        "image_path": str(sample["image_path"]),
        "label_path": str(sample["label_path"]),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    with paths["error"].open("w", encoding="utf-8") as f:
        json.dump(
            row,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 汇总
# ============================================================

def rebuild_prediction_summary():
    rows = []

    for path in sorted(
        OUTPUT_ROOT.glob("*/*/*/*_result.json")
    ):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)

            if row.get("status") == "ok":
                rows.append(row)

        except Exception as exc:
            print(
                f"[警告] 无法读取结果 {path}: {exc}",
                file=sys.stderr,
            )

    jsonl_path = OUTPUT_ROOT / "all_predictions.jsonl"
    csv_path = OUTPUT_ROOT / "all_predictions.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    fields = [
        "image_key",
        "model_name",
        "mode",
        "ground_truth_count",
        "prediction_count",
        "elapsed_seconds",
        "new_tokens",
        "tokens_per_second",
        "reached_max_new_tokens",
        "parse_source",
        "overlay_image",
        "result_json",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )

    return rows, jsonl_path, csv_path


# ============================================================
# 主程序
# ============================================================

def main():
    samples = collect_samples()

    expected_images = 8

    if len(samples) != expected_images:
        print(
            f"[警告] 当前发现 {len(samples)} 张图片，"
            f"DOTA8 通常应为 {expected_images} 张。",
            file=sys.stderr,
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total = len(samples) * len(MODELS) * len(MODES)

    print("=" * 84)
    print("DOTA8 双模型 × 双提示模式 OBB 批量推理")
    print("=" * 84)
    print(f"数据集：{DATASET_ROOT}")
    print(f"图片数量：{len(samples)}")
    print(f"模型：{list(MODELS.keys())}")
    print(f"模式：{list(MODES)}")
    print(f"总推理次数：{total}")
    print(f"结果目录：{OUTPUT_ROOT}")
    print("=" * 84)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    success = 0
    skipped = 0
    failed = 0

    progress = tqdm(
        total=total,
        desc="DOTA8 推理",
        unit="次",
        dynamic_ncols=True,
    )

    try:
        # 模型放在最外层，每个模型只加载一次。
        for model_name, model_path in MODELS.items():
            model, processor = load_model(
                model_name,
                model_path,
            )

            try:
                for image_index, sample in enumerate(
                    samples,
                    start=1,
                ):
                    for mode in MODES:
                        progress.set_postfix(
                            model=model_name,
                            image=(
                                f"{image_index}/{len(samples)}"
                            ),
                            mode=mode,
                        )

                        if (
                            SKIP_EXISTING
                            and is_finished(
                                sample,
                                model_name,
                                mode,
                            )
                        ):
                            skipped += 1
                            progress.update(1)
                            continue

                        try:
                            result = run_one(
                                model=model,
                                processor=processor,
                                model_name=model_name,
                                model_path=model_path,
                                mode=mode,
                                sample=sample,
                            )

                            success += 1

                            progress.set_postfix(
                                model=model_name,
                                image=sample["image_path"].name,
                                mode=mode,
                                gt=result["ground_truth_count"],
                                pred=result["prediction_count"],
                                sec=result["elapsed_seconds"],
                            )

                        except torch.cuda.OutOfMemoryError as exc:
                            failed += 1
                            torch.cuda.empty_cache()

                            save_error(
                                sample,
                                model_name,
                                model_path,
                                mode,
                                exc,
                            )

                            tqdm.write(
                                f"[OOM] {model_name} | "
                                f"{sample['image_key']} | {mode}"
                            )

                        except Exception as exc:
                            failed += 1

                            save_error(
                                sample,
                                model_name,
                                model_path,
                                mode,
                                exc,
                            )

                            tqdm.write(
                                f"[失败] {model_name} | "
                                f"{sample['image_key']} | {mode} | "
                                f"{type(exc).__name__}: {exc}"
                            )

                        finally:
                            progress.update(1)

            finally:
                release_model(
                    model,
                    processor,
                    model_name,
                )

    finally:
        progress.close()

    rows, jsonl_path, csv_path = (
        rebuild_prediction_summary()
    )

    print("\n" + "=" * 84)
    print("批量推理结束")
    print("=" * 84)
    print(f"本次成功：{success}")
    print(f"本次跳过：{skipped}")
    print(f"本次失败：{failed}")
    print(f"有效结果：{len(rows)}/{total}")
    print(f"预测汇总 JSONL：{jsonl_path}")
    print(f"预测汇总 CSV：{csv_path}")
    print(f"结果根目录：{OUTPUT_ROOT}")
    print("=" * 84)


if __name__ == "__main__":
    main()
