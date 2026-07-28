"""Experimental OBB reward. Not part of the strict paper-aligned baseline."""
from __future__ import annotations
import json,re,sys
from pathlib import Path
from typing import Any
sys.path.insert(0,str(Path(__file__).resolve().parent))
try:
    import cv2
    import numpy as np
except Exception as exc:
    raise RuntimeError("Experimental OBB reward requires opencv-python and numpy") from exc
from dota_hbb_reward import norm_class


def parse(response):
    exact=bool(re.search(r"<think>.*?</think>\s*Answer:\s*\{.*?\}\s*$",response,re.S))
    m=re.search(r"Answer:\s*(\{.*?\})\s*$",response,re.S)
    if not m:return None,0.0
    try:v=json.loads(m.group(1))
    except Exception:return None,0.0
    return (v if isinstance(v,dict) else None),(1.0 if exact else 0.5)


def poly_iou(a,b):
    pa=np.asarray(a,dtype=np.float32).reshape(4,2); pb=np.asarray(b,dtype=np.float32).reshape(4,2)
    ca=cv2.convexHull(pa); cb=cv2.convexHull(pb)
    aa=abs(float(cv2.contourArea(ca))); ab=abs(float(cv2.contourArea(cb)))
    inter,_=cv2.intersectConvexConvex(ca,cb); union=aa+ab-max(0,float(inter))
    return max(0,float(inter))/union if union>1e-8 else 0.0


def compute_score(reward_inputs:list[dict[str,Any]],format_weight:float=.1,class_weight:float=.2):
    out=[]
    for item in reward_inputs:
        gt=item.get('ground_truth') or {}; parsed,fmt=parse(str(item.get('response','')))
        cs=1.0 if parsed and norm_class(parsed.get('class_name'))==norm_class(gt.get('class_name')) else 0.0
        obb=parsed.get('obb') if parsed else None
        piou=poly_iou(obb,gt.get('obb')) if isinstance(obb,list) and len(obb)==8 and isinstance(gt.get('obb'),list) else 0.0
        iw=max(0.0,1-format_weight-class_weight); overall=format_weight*fmt+class_weight*cs+iw*piou
        out.append({'overall':float(overall),'format':float(fmt),'class':cs,'OBB_IOU':float(piou)})
    return out
