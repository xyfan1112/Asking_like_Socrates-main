#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估 07_dota8_batch_obb.py 生成的预测结果。

输出：
- metrics_summary.csv / json
- metrics_per_class.csv
- metrics_per_image.csv

主要指标：
- Precision@IoU0.50
- Recall@IoU0.50
- F1@IoU0.50
- mAP50
- mAP50-95
- Count MAE
- Exact Count Accuracy
- Mean IoU of TP
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 配置区
# ============================================================

DATASET_ROOT = Path("/home/yk/fxy/datasets/dota8")
OUTPUT_ROOT = Path("/home/yk/fxy/dota8_vlm_obb_results")

MODELS = ("Qwen", "RS")
MODES = ("direct", "prompt")

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

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp",
}

# P/R/F1 统计时采用的置信度阈值。
CONF_THRESHOLD = 0.25

# mAP50-95。
IOU_THRESHOLDS = [
    round(value, 2)
    for value in np.arange(0.50, 0.96, 0.05)
]


# ============================================================
# 数据读取
# ============================================================

def collect_samples() -> List[Dict]:
    samples = []

    for split in ("train", "val"):
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        for image_path in sorted(image_dir.iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            samples.append(
                {
                    "split": split,
                    "image_path": image_path.resolve(),
                    "label_path": label_path.resolve(),
                    "image_key": f"{split}/{image_path.name}",
                }
            )

    return samples


def order_polygon(
    points: List[List[float]],
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(array) != 4:
        raise ValueError("OBB 必须包含 4 个点")

    center = array.mean(axis=0)

    angles = np.arctan2(
        array[:, 1] - center[1],
        array[:, 0] - center[0],
    )

    order = np.argsort(angles)
    return array[order]


def load_ground_truth(
    sample: Dict,
) -> List[Dict]:
    with Image.open(sample["image_path"]) as image:
        width, height = image.size

    objects = []

    with sample["label_path"].open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 9:
                raise ValueError(
                    f"{sample['label_path']} 第 {line_no} 行格式错误"
                )

            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]

            points = []

            for index in range(0, 8, 2):
                points.append(
                    [
                        values[index] * width,
                        values[index + 1] * height,
                    ]
                )

            objects.append(
                {
                    "image_key": sample["image_key"],
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "points": points,
                }
            )

    return objects


def result_path(
    sample: Dict,
    model_name: str,
    mode: str,
) -> Path:
    return (
        OUTPUT_ROOT
        / model_name
        / mode
        / sample["split"]
        / f"{sample['image_path'].stem}_result.json"
    )


def load_prediction_result(
    sample: Dict,
    model_name: str,
    mode: str,
) -> Tuple[List[Dict], Optional[Dict]]:
    path = result_path(
        sample,
        model_name,
        mode,
    )

    if not path.is_file():
        return [], None

    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if result.get("status") != "ok":
        return [], result

    predictions = []

    for pred in result.get("predictions", []):
        class_id = int(pred["class_id"])

        predictions.append(
            {
                "image_key": sample["image_key"],
                "class_id": class_id,
                "class_name": (
                    CLASS_NAMES[class_id]
                    if 0 <= class_id < len(CLASS_NAMES)
                    else str(pred.get("class_name", class_id))
                ),
                "confidence": float(
                    pred.get("confidence", 0.5)
                ),
                "points": pred["points"],
                "box_format": pred.get(
                    "box_format",
                    "unknown",
                ),
            }
        )

    return predictions, result


# ============================================================
# OBB IoU
# ============================================================

def polygon_area(points: List[List[float]]) -> float:
    polygon = order_polygon(points)
    return float(abs(cv2.contourArea(polygon)))


def obb_iou(
    points_a: List[List[float]],
    points_b: List[List[float]],
) -> float:
    polygon_a = order_polygon(points_a)
    polygon_b = order_polygon(points_b)

    area_a = float(abs(cv2.contourArea(polygon_a)))
    area_b = float(abs(cv2.contourArea(polygon_b)))

    if area_a <= 1e-8 or area_b <= 1e-8:
        return 0.0

    try:
        intersection_area, _ = cv2.intersectConvexConvex(
            polygon_a,
            polygon_b,
        )
    except cv2.error:
        hull_a = cv2.convexHull(polygon_a)
        hull_b = cv2.convexHull(polygon_b)

        intersection_area, _ = cv2.intersectConvexConvex(
            hull_a,
            hull_b,
        )

    intersection_area = max(
        0.0,
        float(intersection_area),
    )

    union = area_a + area_b - intersection_area

    if union <= 1e-8:
        return 0.0

    return intersection_area / union


# ============================================================
# AP 和配对匹配
# ============================================================

def compute_ap_101(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """
    COCO 风格 101 点插值 AP。
    """

    if recalls.size == 0:
        return 0.0

    recall_points = np.linspace(0.0, 1.0, 101)

    ap_values = []

    for recall_level in recall_points:
        candidates = precisions[recalls >= recall_level]

        if candidates.size:
            ap_values.append(float(candidates.max()))
        else:
            ap_values.append(0.0)

    return float(np.mean(ap_values))


def evaluate_class_ap(
    ground_truth: List[Dict],
    predictions: List[Dict],
    class_id: int,
    iou_threshold: float,
) -> Dict:
    class_gt = [
        obj
        for obj in ground_truth
        if obj["class_id"] == class_id
    ]

    class_pred = [
        obj
        for obj in predictions
        if obj["class_id"] == class_id
    ]

    gt_by_image = defaultdict(list)

    for obj in class_gt:
        gt_by_image[obj["image_key"]].append(obj)

    class_pred = sorted(
        class_pred,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    matched = {
        image_key: set()
        for image_key in gt_by_image
    }

    tp = []
    fp = []

    for pred in class_pred:
        candidates = gt_by_image.get(
            pred["image_key"],
            [],
        )

        best_iou = 0.0
        best_index = None

        for gt_index, gt in enumerate(candidates):
            if gt_index in matched[pred["image_key"]]:
                continue

            iou = obb_iou(
                pred["points"],
                gt["points"],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = gt_index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):
            tp.append(1)
            fp.append(0)
            matched[pred["image_key"]].add(best_index)
        else:
            tp.append(0)
            fp.append(1)

    number_gt = len(class_gt)

    if not class_pred:
        return {
            "ap": 0.0,
            "number_gt": number_gt,
            "number_pred": 0,
        }

    tp_cumulative = np.cumsum(
        np.asarray(tp, dtype=np.float64)
    )

    fp_cumulative = np.cumsum(
        np.asarray(fp, dtype=np.float64)
    )

    recalls = (
        tp_cumulative / max(number_gt, 1)
    )

    precisions = (
        tp_cumulative
        / np.maximum(
            tp_cumulative + fp_cumulative,
            1e-12,
        )
    )

    ap = compute_ap_101(
        recalls,
        precisions,
    )

    return {
        "ap": ap,
        "number_gt": number_gt,
        "number_pred": len(class_pred),
    }


def greedy_match_image(
    gt_objects: List[Dict],
    pred_objects: List[Dict],
    iou_threshold: float,
    confidence_threshold: float,
) -> Dict:
    predictions = [
        pred
        for pred in pred_objects
        if pred["confidence"] >= confidence_threshold
    ]

    predictions = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    matched_gt = set()
    matched_ious = []
    tp = 0
    fp = 0

    for pred in predictions:
        best_iou = 0.0
        best_index = None

        for gt_index, gt in enumerate(gt_objects):
            if gt_index in matched_gt:
                continue

            if gt["class_id"] != pred["class_id"]:
                continue

            iou = obb_iou(
                pred["points"],
                gt["points"],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = gt_index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):
            tp += 1
            matched_gt.add(best_index)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_objects) - len(matched_gt)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_iou = (
        float(np.mean(matched_ious))
        if matched_ious
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_tp": mean_iou,
        "matched_ious": matched_ious,
        "pred_count_above_conf": len(predictions),
    }


# ============================================================
# 单个 model × mode 评估
# ============================================================

def evaluate_condition(
    samples: List[Dict],
    model_name: str,
    mode: str,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    all_gt = []
    all_predictions = []
    image_rows = []
    result_rows = {}

    missing_results = []

    for sample in samples:
        gt_objects = load_ground_truth(sample)

        predictions, result = load_prediction_result(
            sample,
            model_name,
            mode,
        )

        if result is None:
            missing_results.append(sample["image_key"])

        all_gt.extend(gt_objects)
        all_predictions.extend(predictions)

        result_rows[sample["image_key"]] = result

        matched = greedy_match_image(
            gt_objects,
            predictions,
            iou_threshold=0.50,
            confidence_threshold=CONF_THRESHOLD,
        )

        gt_count = len(gt_objects)
        pred_count = len(predictions)

        image_rows.append(
            {
                "model_name": model_name,
                "mode": mode,
                "image_key": sample["image_key"],
                "gt_count": gt_count,
                "pred_count": pred_count,
                "count_absolute_error": abs(
                    pred_count - gt_count
                ),
                "count_exact": int(
                    pred_count == gt_count
                ),
                "tp50": matched["tp"],
                "fp50": matched["fp"],
                "fn50": matched["fn"],
                "precision50": matched["precision"],
                "recall50": matched["recall"],
                "f1_50": matched["f1"],
                "mean_iou_tp50": matched["mean_iou_tp"],
                "missing_result": int(result is None),
                "reached_max_new_tokens": (
                    result.get(
                        "reached_max_new_tokens",
                        False,
                    )
                    if result
                    else None
                ),
                "elapsed_seconds": (
                    result.get("elapsed_seconds")
                    if result
                    else None
                ),
            }
        )

    classes_with_gt = sorted(
        {
            obj["class_id"]
            for obj in all_gt
        }
    )

    class_rows = []
    ap_by_threshold = defaultdict(list)

    for class_id in classes_with_gt:
        ap_values = []

        number_gt = sum(
            obj["class_id"] == class_id
            for obj in all_gt
        )

        number_pred = sum(
            obj["class_id"] == class_id
            for obj in all_predictions
        )

        for iou_threshold in IOU_THRESHOLDS:
            result = evaluate_class_ap(
                all_gt,
                all_predictions,
                class_id,
                iou_threshold,
            )

            ap_values.append(result["ap"])
            ap_by_threshold[iou_threshold].append(
                result["ap"]
            )

        class_rows.append(
            {
                "model_name": model_name,
                "mode": mode,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "gt_count": number_gt,
                "pred_count": number_pred,
                "ap50": ap_values[0],
                "map50_95": float(
                    np.mean(ap_values)
                ),
            }
        )

    map50 = (
        float(np.mean(ap_by_threshold[0.50]))
        if ap_by_threshold[0.50]
        else 0.0
    )

    map50_95 = (
        float(
            np.mean(
                [
                    ap
                    for threshold
                    in IOU_THRESHOLDS
                    for ap
                    in ap_by_threshold[threshold]
                ]
            )
        )
        if classes_with_gt
        else 0.0
    )

    aggregate_tp = sum(row["tp50"] for row in image_rows)
    aggregate_fp = sum(row["fp50"] for row in image_rows)
    aggregate_fn = sum(row["fn50"] for row in image_rows)

    precision = (
        aggregate_tp / (aggregate_tp + aggregate_fp)
        if aggregate_tp + aggregate_fp > 0
        else 0.0
    )

    recall = (
        aggregate_tp / (aggregate_tp + aggregate_fn)
        if aggregate_tp + aggregate_fn > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    all_matched_ious = []

    for sample in samples:
        gt_objects = load_ground_truth(sample)
        predictions, _ = load_prediction_result(
            sample,
            model_name,
            mode,
        )

        match = greedy_match_image(
            gt_objects,
            predictions,
            iou_threshold=0.50,
            confidence_threshold=CONF_THRESHOLD,
        )

        all_matched_ious.extend(
            match["matched_ious"]
        )

    gt_total = len(all_gt)
    pred_total = len(all_predictions)

    count_mae = float(
        np.mean(
            [
                row["count_absolute_error"]
                for row in image_rows
            ]
        )
    )

    exact_count_accuracy = float(
        np.mean(
            [
                row["count_exact"]
                for row in image_rows
            ]
        )
    )

    summary = {
        "model_name": model_name,
        "mode": mode,
        "images": len(samples),
        "missing_results": len(missing_results),
        "missing_image_keys": missing_results,
        "classes_with_gt": len(classes_with_gt),
        "gt_total": gt_total,
        "pred_total": pred_total,
        "confidence_threshold_for_prf1": CONF_THRESHOLD,
        "tp50": aggregate_tp,
        "fp50": aggregate_fp,
        "fn50": aggregate_fn,
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
        "map50": map50,
        "map50_95": map50_95,
        "count_mae": count_mae,
        "exact_count_accuracy": exact_count_accuracy,
        "mean_iou_tp50": (
            float(np.mean(all_matched_ious))
            if all_matched_ious
            else 0.0
        ),
    }

    return summary, class_rows, image_rows


# ============================================================
# 保存
# ============================================================

def write_csv(
    path: Path,
    rows: List[Dict],
    fieldnames: List[str],
):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fieldnames
                }
            )


def main():
    samples = collect_samples()

    summaries = []
    class_rows_all = []
    image_rows_all = []

    for model_name in MODELS:
        for mode in MODES:
            (
                summary,
                class_rows,
                image_rows,
            ) = evaluate_condition(
                samples,
                model_name,
                mode,
            )

            summaries.append(summary)
            class_rows_all.extend(class_rows)
            image_rows_all.extend(image_rows)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_json = (
        OUTPUT_ROOT
        / "metrics_summary.json"
    )

    summary_csv = (
        OUTPUT_ROOT
        / "metrics_summary.csv"
    )

    per_class_csv = (
        OUTPUT_ROOT
        / "metrics_per_class.csv"
    )

    per_image_csv = (
        OUTPUT_ROOT
        / "metrics_per_image.csv"
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summaries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary_fields = [
        "model_name",
        "mode",
        "images",
        "missing_results",
        "classes_with_gt",
        "gt_total",
        "pred_total",
        "confidence_threshold_for_prf1",
        "tp50",
        "fp50",
        "fn50",
        "precision50",
        "recall50",
        "f1_50",
        "map50",
        "map50_95",
        "count_mae",
        "exact_count_accuracy",
        "mean_iou_tp50",
    ]

    write_csv(
        summary_csv,
        summaries,
        summary_fields,
    )

    write_csv(
        per_class_csv,
        class_rows_all,
        [
            "model_name",
            "mode",
            "class_id",
            "class_name",
            "gt_count",
            "pred_count",
            "ap50",
            "map50_95",
        ],
    )

    write_csv(
        per_image_csv,
        image_rows_all,
        [
            "model_name",
            "mode",
            "image_key",
            "gt_count",
            "pred_count",
            "count_absolute_error",
            "count_exact",
            "tp50",
            "fp50",
            "fn50",
            "precision50",
            "recall50",
            "f1_50",
            "mean_iou_tp50",
            "missing_result",
            "reached_max_new_tokens",
            "elapsed_seconds",
        ],
    )

    print("=" * 112)
    print("DOTA8 VLM OBB 评估结果")
    print("=" * 112)

    header = (
        f"{'Model':8s} "
        f"{'Mode':8s} "
        f"{'GT':>5s} "
        f"{'Pred':>6s} "
        f"{'P50':>8s} "
        f"{'R50':>8s} "
        f"{'F1':>8s} "
        f"{'mAP50':>8s} "
        f"{'mAP50-95':>11s} "
        f"{'CountMAE':>10s} "
        f"{'Missing':>8s}"
    )

    print(header)
    print("-" * len(header))

    for row in summaries:
        print(
            f"{row['model_name']:8s} "
            f"{row['mode']:8s} "
            f"{row['gt_total']:5d} "
            f"{row['pred_total']:6d} "
            f"{row['precision50'] * 100:8.2f} "
            f"{row['recall50'] * 100:8.2f} "
            f"{row['f1_50'] * 100:8.2f} "
            f"{row['map50'] * 100:8.2f} "
            f"{row['map50_95'] * 100:11.2f} "
            f"{row['count_mae']:10.2f} "
            f"{row['missing_results']:8d}"
        )

    print("=" * 112)
    print(f"汇总 JSON：{summary_json}")
    print(f"汇总 CSV：{summary_csv}")
    print(f"逐类别 CSV：{per_class_csv}")
    print(f"逐图片 CSV：{per_image_csv}")
    print("=" * 112)


if __name__ == "__main__":
    main()
