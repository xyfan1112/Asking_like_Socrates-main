"""Batch reward for EasyR1: format + canonical class + normalized HBB IoU."""
from __future__ import annotations
import json, math, re
from typing import Any

DOTA_CLASSES = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
}
ALIASES = {
    "bus":"large vehicle", "truck":"large vehicle", "lorry":"large vehicle",
    "trailer":"large vehicle", "car":"small vehicle", "sedan":"small vehicle",
    "suv":"small vehicle", "aircraft":"plane", "airplane":"plane",
    "boat":"ship", "vessel":"ship",
}


def norm_class(value: Any) -> str | None:
    x = re.sub(r"\s+", " ", str(value or "").strip().lower().replace("_", " ").replace("-", " "))
    x = ALIASES.get(x, x)
    return x if x in DOTA_CLASSES else None


def parse_answer(response: str) -> tuple[dict | None, float]:
    exact = bool(re.search(r"<think>.*?</think>\s*Answer:\s*\{.*?\}\s*$", response, re.S))
    match = re.search(r"Answer:\s*(\{.*?\})\s*$", response, re.S)
    if not match:
        return None, 0.0
    try:
        value = json.loads(match.group(1))
    except Exception:
        return None, 0.0
    return value if isinstance(value, dict) else None, 1.0 if exact else 0.5


def iou(a, b):
    ax1,ay1,ax2,ay2=[float(x) for x in a]; bx1,by1,bx2,by2=[float(x) for x in b]
    ax1,ax2=sorted((max(0,min(1000,ax1)),max(0,min(1000,ax2))))
    ay1,ay2=sorted((max(0,min(1000,ay1)),max(0,min(1000,ay2))))
    bx1,bx2=sorted((bx1,bx2)); by1,by2=sorted((by1,by2))
    inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float=0.1, class_weight: float=0.2) -> list[dict[str,float]]:
    results=[]
    for item in reward_inputs:
        gt=item.get("ground_truth") or {}
        parsed, fmt=parse_answer(str(item.get("response", "")))
        pred_class=norm_class(parsed.get("class_name")) if parsed else None
        gt_class=norm_class(gt.get("class_name"))
        class_score=1.0 if pred_class and pred_class==gt_class else 0.0
        box=parsed.get("bbox") if parsed else None
        box_iou=iou(box,gt.get("bbox")) if isinstance(box,list) and len(box)==4 and isinstance(gt.get("bbox"),list) else 0.0
        accuracy_weight=max(0.0,1.0-format_weight-class_weight)
        overall=format_weight*fmt+class_weight*class_score+accuracy_weight*box_iou
        results.append({"overall":float(overall),"format":float(fmt),"class":class_score,"IOU":float(box_iou)})
    return results
