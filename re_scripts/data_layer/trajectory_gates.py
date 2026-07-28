"""Deterministic class, geometry, and trace-consistency gates.

The final exact OBB may be teacher-forced from GT, but the generated dialogue must
still identify the right class and image region and contain usable coarse spatial
evidence. The gate therefore searches both the final answer and Perceiver turns.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from main_layer.common import (
    extract_final_text,
    hbb_from_points,
    hbb_iou,
    infer_and_convert_points,
    normalize_class_name,
    parse_obb_output,
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BBOX_RE = re.compile(
    r"(?:bbox_2d|box_2d|bounding\s*box)\s*(?:=|:)?\s*\[([^\]]+)\]",
    re.IGNORECASE,
)
_OBB_RE = re.compile(
    r"(?:obb_8|obb|corners?)\s*(?:=|:)?\s*\[([^\]]+)\]",
    re.IGNORECASE,
)
_FOCUS_RE = re.compile(
    r"(?:focus\s+region|roi)[^\[\n]{0,80}\[\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE,
)

CLASS_EVIDENCE = {
    "plane": ("plane", "aircraft", "airplane", "wings", "fuselage", "tail section"),
    "helicopter": ("helicopter", "rotor", "rotary-wing"),
    "ship": ("ship", "boat", "vessel", "watercraft", "barge"),
    "harbor": ("harbor", "marina", "dock", "pier", "waterfront facility"),
    "storage tank": ("storage tank", "tank", "cylindrical", "circular industrial"),
    "bridge": ("bridge", "crossing", "spans the"),
    "large vehicle": ("large vehicle", "truck", "bus", "trailer", "lorry", "semi-truck"),
    "small vehicle": ("small vehicle", "car", "sedan", "suv", "pickup"),
    "roundabout": ("roundabout", "circular intersection", "central island"),
    "baseball diamond": ("baseball diamond", "baseball field", "infield", "diamond-shaped"),
    "tennis court": ("tennis court", "service boxes", "tennis"),
    "basketball court": ("basketball court", "basketball"),
    "ground track field": ("ground track field", "running track", "oval track"),
    "soccer ball field": ("soccer", "football field", "soccer field"),
    "swimming pool": ("swimming pool", "pool", "water facility"),
}

CLASS_CLAIM_TERMS = {
    "plane": ("plane", "aircraft", "airplane", "jet"),
    "helicopter": ("helicopter", "rotorcraft"),
    "ship": ("ship", "boat", "vessel", "watercraft"),
    "harbor": ("harbor", "marina", "port", "dockyard"),
    "storage tank": ("storage tank", "tank"),
    "bridge": ("bridge",),
    "large vehicle": ("large vehicle", "truck", "bus", "trailer", "lorry"),
    "small vehicle": ("small vehicle", "car", "sedan", "suv", "pickup"),
    "roundabout": ("roundabout", "traffic circle"),
    "baseball diamond": ("baseball diamond", "baseball field"),
    "tennis court": ("tennis court",),
    "basketball court": ("basketball court",),
    "ground track field": ("ground track field", "running track"),
    "soccer ball field": ("soccer ball field", "soccer field", "football field"),
    "swimming pool": ("swimming pool",),
}

SCENE_REGION_TERMS = (
    "parking lot",
    "airport",
    "airfield",
    "runway",
    "marina",
    "dock area",
    "docking area",
    "industrial building",
    "warehouse",
    "building",
    "road",
    "waterway",
    "industrial area",
    "commercial area",
    "residential area",
)
SCENE_COMPATIBLE_BY_CLASS = {
    "harbor": {"marina", "dock area", "docking area"},
}
_QUESTION_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "any",
    "are",
    "be",
    "can",
    "could",
    "do",
    "does",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "present",
    "that",
    "the",
    "there",
    "this",
    "to",
    "what",
    "which",
    "with",
}


def coordinate_parse_mode(target: str | None) -> str:
    return {
        "pixel_obb": "pixel",
        "norm100_obb": "normalized_0_100",
        "norm1000_obb": "normalized_0_1000",
    }.get(target, "auto")


def expanded_center_gate(pred_hbb: list[float], gt_hbb: list[float], ratio: float) -> bool:
    px = (pred_hbb[0] + pred_hbb[2]) / 2
    py = (pred_hbb[1] + pred_hbb[3]) / 2
    gx1, gy1, gx2, gy2 = gt_hbb
    margin_x = max(1.0, (gx2 - gx1) * ratio)
    margin_y = max(1.0, (gy2 - gy1) * ratio)
    return gx1 - margin_x <= px <= gx2 + margin_x and gy1 - margin_y <= py <= gy2 + margin_y


def _nums(block: str) -> list[float]:
    return [float(x) for x in _NUMBER_RE.findall(block)]


def _points_from_values(
    values: list[float], width: int, height: int, mode: str, kind: str
) -> tuple[list[list[float]], str] | None:
    if kind == "bbox" and len(values) >= 4:
        x1, y1, x2, y2 = values[:4]
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif kind == "obb" and len(values) >= 8:
        pts = [[values[i], values[i + 1]] for i in range(0, 8, 2)]
    else:
        return None
    converted, detected = infer_and_convert_points(pts, width, height, mode)
    return converted, detected


def extract_spatial_candidates(
    text: str | None,
    width: int,
    height: int,
    coord_mode: str,
    source: str,
) -> list[dict[str, Any]]:
    """Extract OBB/HBB candidates from a final answer or a Perceiver response."""
    content = str(text or "")
    candidates: list[dict[str, Any]] = []

    for pred in parse_obb_output(content, width, height, coord_mode):
        candidates.append({**pred, "source": source, "kind": "class_obb_line"})

    for match in _BBOX_RE.finditer(content):
        parsed = _points_from_values(_nums(match.group(1)), width, height, coord_mode, "bbox")
        if parsed:
            points, detected = parsed
            candidates.append(
                {
                    "class_name": None,
                    "points": points,
                    "coordinate_mode_detected": detected,
                    "source": source,
                    "kind": "bbox_2d",
                }
            )
    for match in _OBB_RE.finditer(content):
        parsed = _points_from_values(_nums(match.group(1)), width, height, coord_mode, "obb")
        if parsed:
            points, detected = parsed
            candidates.append(
                {
                    "class_name": None,
                    "points": points,
                    "coordinate_mode_detected": detected,
                    "source": source,
                    "kind": "obb_8",
                }
            )
    return candidates


def _base_result(task: str | None, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "task": task,
        "class_ok": False,
        "parse_ok": False,
        "hbb_iou": 0.0,
        "iou_gate": False,
        "center_gate": False,
        "pass": False,
        "reason": "not_evaluated",
    }


def audit_generated_answer(
    item: dict[str, Any], generated_answer: str, settings: dict[str, Any]
) -> dict[str, Any]:
    """Audit the generated final answer only (used for classification and diagnostics)."""
    gate = settings["trajectory"].get("geometry_gate", {})
    task = item.get("task")
    generated = extract_final_text(generated_answer)
    gt_class = normalize_class_name(item.get("class_name") or str(item.get("gt", "")).split("|", 1)[0])
    result = _base_result(task, bool(gate.get("enabled", True)))
    if not result["enabled"]:
        result.update({"pass": True, "reason": "disabled"})
        return result

    if task == "ref_classification":
        pred_class = normalize_class_name(generated)
        result.update(
            {
                "pred_class": pred_class,
                "gt_class": gt_class,
                "class_ok": pred_class == gt_class and pred_class is not None,
                "parse_ok": pred_class is not None,
            }
        )
        result["pass"] = result["class_ok"]
        result["reason"] = "ok" if result["pass"] else "canonical_class_mismatch"
        return result

    if task != "ref_grounding_obb":
        result.update({"pass": True, "reason": "non_ref_task"})
        return result

    width, height = int(item.get("image_width", 0)), int(item.get("image_height", 0))
    gt_points = item.get("obb_pixel") or item.get("obj_corner_pixel")
    if width <= 0 or height <= 0 or not isinstance(gt_points, list) or len(gt_points) != 4:
        result["reason"] = "missing_image_or_gt"
        return result
    predictions = parse_obb_output(generated, width, height, coordinate_parse_mode(item.get("coordinate_target")))
    if not predictions:
        result["reason"] = "no_parseable_obb"
        return result
    return _score_candidates(item, predictions, settings, require_class=True, source_mode="final_only")


def _polygon_area(points: list[list[float]]) -> float:
    return abs(sum(
        points[i][0] * points[(i + 1) % 4][1]
        - points[(i + 1) % 4][0] * points[i][1]
        for i in range(4)
    )) / 2.0


def _orient(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _crosses(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    eps=1e-8
    return _orient(a,b,c)*_orient(a,b,d)<-eps and _orient(c,d,a)*_orient(c,d,b)<-eps


def validate_quadrilateral(points: Any) -> tuple[bool,str]:
    if not isinstance(points,list) or len(points)!=4:
        return False,'not_four_points'
    try:
        pts=[[float(p[0]),float(p[1])] for p in points]
    except Exception:
        return False,'non_numeric_points'
    if len({(round(p[0],4),round(p[1],4)) for p in pts})<4:
        return False,'duplicate_points'
    if _polygon_area(pts)<1.0:
        return False,'zero_or_tiny_area'
    if _crosses(pts[0],pts[1],pts[2],pts[3]) or _crosses(pts[1],pts[2],pts[3],pts[0]):
        return False,'self_intersection'
    return True,'ok'


def _focus_hbb_pixel(item: dict[str,Any], width:int, height:int) -> list[float] | None:
    match=_FOCUS_RE.search(str(item.get('query') or item.get('question') or ''))
    if not match:
        return None
    x1,y1,x2,y2=[float(v) for v in match.groups()]
    if x2<=x1 or y2<=y1:
        return None
    return [x1*width/1000.0,y1*height/1000.0,x2*width/1000.0,y2*height/1000.0]


def _candidate_quality(
    item: dict[str,Any], candidate: dict[str,Any], gt_hbb:list[float],
    width:int, height:int, settings:dict[str,Any],
) -> dict[str,Any]:
    points=candidate.get('points')
    valid,invalid_reason=validate_quadrilateral(points)
    pred_hbb=hbb_from_points(points) if isinstance(points,list) and len(points)==4 else [0,0,0,0]
    focus_hbb=_focus_hbb_pixel(item,width,height)
    focus_iou=hbb_iou(pred_hbb,focus_hbb) if focus_hbb else 0.0
    gt_area=max(1.0,(gt_hbb[2]-gt_hbb[0])*(gt_hbb[3]-gt_hbb[1]))
    pred_area=max(0.0,(pred_hbb[2]-pred_hbb[0])*(pred_hbb[3]-pred_hbb[1]))
    area_ratio=pred_area/gt_area
    cfg=settings.get('trajectory',{})
    reject_focus=bool(cfg.get('strict_reject_focus_copy',True))
    focus_copy=bool(focus_hbb and focus_iou>=0.92)
    max_ratio=float(cfg.get('strict_max_pred_gt_hbb_area_ratio',6.0))
    quality_ok=valid and (not reject_focus or not focus_copy) and area_ratio<=max_ratio
    reason='ok'
    if not valid: reason=invalid_reason
    elif reject_focus and focus_copy: reason='copied_coarse_focus_roi'
    elif area_ratio>max_ratio: reason=f'pred_gt_area_ratio_too_large:{area_ratio:.3f}'
    return {
        'quality_ok':quality_ok,'quality_reason':reason,'quadrilateral_valid':valid,
        'focus_copy':focus_copy,'focus_iou':round(float(focus_iou),6),
        'pred_gt_hbb_area_ratio':round(float(area_ratio),6),'pred_hbb':pred_hbb,
    }


def _score_candidates(
    item: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    settings: dict[str, Any],
    *,
    require_class: bool,
    source_mode: str,
) -> dict[str, Any]:
    gate = settings["trajectory"].get("geometry_gate", {})
    task = item.get("task")
    result = _base_result(task, bool(gate.get("enabled", True)))
    gt_class = normalize_class_name(item.get("class_name") or str(item.get("gt", "")).split("|", 1)[0])
    gt_points = item.get("obb_pixel") or item.get("obj_corner_pixel")
    if not isinstance(gt_points, list) or len(gt_points) != 4:
        result["reason"] = "missing_gt_obb"
        return result
    gt_hbb = hbb_from_points(gt_points)

    width, height = int(item.get("image_width", 0)), int(item.get("image_height", 0))
    best: tuple[float, dict[str, Any], list[float], dict[str,Any]] | None = None
    rejected_quality: list[dict[str,Any]] = []
    for candidate in candidates:
        points = candidate.get("points")
        if not isinstance(points, list) or len(points) != 4:
            continue
        quality = _candidate_quality(item,candidate,gt_hbb,width,height,settings)
        if not quality['quality_ok']:
            rejected_quality.append({
                'source':candidate.get('source'),'kind':candidate.get('kind'),
                'reason':quality['quality_reason'],
            })
            continue
        pred_hbb = quality['pred_hbb']
        score = hbb_iou(pred_hbb, gt_hbb)
        if best is None or score > best[0]:
            best = (score, candidate, pred_hbb, quality)
    if best is None:
        result.update({
            "reason": "no_valid_spatial_evidence",
            "rejected_spatial_candidates": rejected_quality,
            "quadrilateral_valid": False,
            "focus_copy": any(r.get('reason')=='copied_coarse_focus_roi' for r in rejected_quality),
        })
        return result

    score, pred, pred_hbb, quality = best
    pred_class = pred.get("class_name")
    class_ok = pred_class == gt_class if pred_class else not require_class
    center_ok = expanded_center_gate(
        pred_hbb, gt_hbb, float(gate.get("expanded_gt_ratio", 0.35))
    )
    min_iou = float(gate.get("min_hbb_iou_for_teacher_force", 0.1))
    require_center = bool(gate.get("require_center_in_expanded_gt", True))
    pass_if_iou_or_center = bool(gate.get("pass_if_iou_or_center", False))
    iou_ok = score >= min_iou
    geometry_ok = (
        (iou_ok or center_ok)
        if pass_if_iou_or_center
        else iou_ok and (center_ok or not require_center)
    )
    if not require_center and pass_if_iou_or_center:
        geometry_ok = score >= min_iou
    passed = class_ok and geometry_ok
    result.update(
        {
            "parse_ok": True,
            "class_ok": class_ok,
            "pred_class": pred_class,
            "gt_class": gt_class,
            "hbb_iou": round(float(score), 6),
            "iou_gate": bool(iou_ok),
            "center_gate": center_ok,
            "pred_points_pixel": pred.get("points"),
            "pred_coordinate_mode": pred.get("coordinate_mode_detected"),
            "evidence_source": pred.get("source"),
            "evidence_kind": pred.get("kind"),
            "source_mode": source_mode,
            "quadrilateral_valid": quality.get("quadrilateral_valid"),
            "focus_copy": quality.get("focus_copy"),
            "focus_iou": quality.get("focus_iou"),
            "pred_gt_hbb_area_ratio": quality.get("pred_gt_hbb_area_ratio"),
            "rejected_spatial_candidates": rejected_quality,
            "pass": passed,
            "reason": "ok"
            if passed
            else (
                "canonical_class_mismatch"
                if not class_ok
                else "coarse_geometry_mismatch"
            ),
        }
    )
    return result


def audit_trajectory_evidence(
    item: dict[str, Any], raw: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """Use the best final/Perceiver coordinate evidence for teacher-forcing gates."""
    if item.get("task") != "ref_grounding_obb":
        return audit_generated_answer(
            item, str((raw.get("loop_result") or {}).get("final_answer") or ""), settings
        )
    width, height = int(item.get("image_width", 0)), int(item.get("image_height", 0))
    if width <= 0 or height <= 0:
        return {**_base_result(item.get("task"), True), "reason": "missing_image_size"}
    mode = coordinate_parse_mode(item.get("coordinate_target"))
    loop = raw.get("loop_result") if isinstance(raw.get("loop_result"), dict) else {}
    final = str(loop.get("final_answer") or "")
    final_candidates = extract_spatial_candidates(final, width, height, mode, "final_answer")
    candidates = list(final_candidates)
    for turn in loop.get("chat_history") or []:
        if turn.get("P_response"):
            candidates.extend(
                extract_spatial_candidates(
                    str(turn["P_response"]),
                    width,
                    height,
                    mode,
                    f"perceiver_round_{turn.get('round', '?')}",
                )
            )
    result = _score_candidates(
        item, candidates, settings, require_class=False, source_mode="trace_evidence"
    )
    gt_class=normalize_class_name(item.get('class_name') or str(item.get('gt','')).split('|',1)[0])
    final_valid=False; final_class_ok=False; final_reason='no_parseable_final_obb'; final_focus_copy=False
    gt_points=item.get('obb_pixel') or item.get('obj_corner_pixel')
    gt_hbb=hbb_from_points(gt_points) if isinstance(gt_points,list) and len(gt_points)==4 else [0,0,0,0]
    for candidate in final_candidates:
        quality=_candidate_quality(item,candidate,gt_hbb,width,height,settings)
        class_ok=normalize_class_name(candidate.get('class_name'))==gt_class
        if quality['focus_copy']: final_focus_copy=True
        if quality['quality_ok'] and class_ok:
            final_valid=True; final_class_ok=True; final_reason='ok'; break
        if not class_ok: final_reason='final_canonical_class_mismatch'
        else: final_reason=quality['quality_reason']
    strict_require=bool(settings.get('trajectory',{}).get('strict_require_valid_obb',True))
    result.update({
        'final_obb_valid':final_valid,
        'final_class_ok':final_class_ok,
        'final_obb_reason':final_reason,
        'final_focus_copy':final_focus_copy,
    })
    if strict_require and not final_valid:
        result['pass']=False
        result['reason']=final_reason
    return result


def _last_evidence_text(loop: dict[str, Any], turns: int = 2) -> str:
    history = loop.get("chat_history") if isinstance(loop.get("chat_history"), list) else []
    chunks: list[str] = []
    for turn in history[-turns:]:
        chunks.append(str(turn.get("R_response") or ""))
        chunks.append(str(turn.get("P_response") or ""))
    return " ".join(chunks).lower()


def _question_tokens(question: str) -> list[str]:
    return [
        token
        for token in _QUESTION_TOKEN_RE.findall(str(question or "").lower())
        if token not in _QUESTION_STOPWORDS
    ]


def _question_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _question_tokens(left), _question_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_norm, right_norm = " ".join(left_tokens), " ".join(right_tokens)
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_set, right_set = set(left_tokens), set(right_tokens)
    jaccard = len(left_set & right_set) / max(1, len(left_set | right_set))
    return max(float(sequence), float(jaccard))


def _reasoner_questions(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in history:
        response = str(turn.get("R_response") or "")
        matches = re.findall(
            r"<question>\s*([\s\S]*?)\s*</question>",
            response,
            flags=re.IGNORECASE,
        )
        for question in matches:
            rows.append(
                {
                    "round": turn.get("round"),
                    "question": re.sub(r"\s+", " ", question).strip(),
                }
            )
    return rows


def _duplicate_question_pairs(
    questions: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    duplicates: list[dict[str, Any]] = []
    for right_index in range(1, len(questions)):
        for left_index in range(right_index):
            score = _question_similarity(
                questions[left_index]["question"],
                questions[right_index]["question"],
            )
            if score >= threshold:
                duplicates.append(
                    {
                        "left_round": questions[left_index]["round"],
                        "right_round": questions[right_index]["round"],
                        "similarity": round(score, 4),
                        "left": questions[left_index]["question"],
                        "right": questions[right_index]["question"],
                    }
                )
                break
    return duplicates


def _classification_leading_questions(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    terms = sorted(
        {
            term
            for values in CLASS_CLAIM_TERMS.values()
            for term in values
        },
        key=len,
        reverse=True,
    )
    bad: list[dict[str, Any]] = []
    for row in questions:
        low = row["question"].lower()
        generic = bool(
            re.search(
                r"\b(?:canonical\s+dota\s+)?(?:category|class|label)\b",
                low,
            )
        )
        revealed = any(
            re.search(rf"\b{re.escape(term)}s?\b", low)
            for term in terms
        )
        if generic or revealed:
            bad.append(row)
    return bad


def _target_denials(perception_texts: list[str], target: str) -> list[str]:
    aliases = CLASS_CLAIM_TERMS.get(target, (target,))
    denied: list[str] = []
    for text in perception_texts:
        low = text.lower()
        for alias in aliases:
            patterns = (
                rf"\bno\s+(?:such\s+|clear\s+)?{re.escape(alias)}s?\b",
                rf"\bnot\s+(?:an?\s+|the\s+)?{re.escape(alias)}\b",
                rf"\bwithout\s+(?:an?\s+|any\s+)?{re.escape(alias)}s?\b",
                rf"\b(?:does|do)\s+not\s+(?:appear|look|seem)[^.!?]{{0,35}}\b{re.escape(alias)}\b",
            )
            if any(re.search(pattern, low) for pattern in patterns):
                denied.append(text)
                break
    return denied


def _strong_class_claims(perception_texts: list[str]) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    subject = (
        r"(?:\btarget\b|\bobject\b|\bstructure\b|\bfeature\b|\bit\b)"
    )
    attribution = (
        subject
        + r"[^.!?]{0,100}?"
        + r"(?:\bis\b|\bare\b|\bappears?\s+to\s+be\b|\blooks?\s+like\b|"
        r"\bseems?\s+to\s+be\b|\bidentified\s+as\b|\bclassified\s+as\b|"
        r"\b(?:could\s+be\s+)?interpreted\s+as\b|"
        r"\b(?:appears?|seems?)\s+consistent\s+with\b|"
        r"\b(?:appears?|seems?)\s+to\s+fit\s+(?:the\s+)?description\s+of\b|"
        r"\bfits?\s+(?:the\s+)?description\s+of\b|"
        r"\bis\s+indicative\s+of\b)"
        r"\s+(?:an?\s+|the\s+)?"
    )
    for text in perception_texts:
        low = text.lower()
        for class_name, aliases in CLASS_CLAIM_TERMS.items():
            for alias in aliases:
                for match in re.finditer(attribution + rf"{re.escape(alias)}s?\b", low):
                    prefix = low[max(0, match.start() - 25) : match.start()]
                    if re.search(r"\b(?:no|not|without|isn['’]?t|aren['’]?t)\b", prefix):
                        continue
                    claims.setdefault(class_name, []).append(text)
                    break
                if class_name in claims and text in claims[class_name]:
                    break
    return claims


def _scene_substitutions(
    perception_texts: list[str],
    target: str,
) -> list[dict[str, str]]:
    compatible = SCENE_COMPATIBLE_BY_CLASS.get(target, set())
    substitutions: list[dict[str, str]] = []
    for text in perception_texts:
        low = text.lower()
        for term in SCENE_REGION_TERMS:
            if term in compatible:
                continue
            pattern = (
                r"(?:\btarget\b|\bobject\b|\bstructure\b|\bfeature\b|\bit\b)"
                r"[^.!?]{0,70}?"
                r"(?:\bis\b|\bappears?\s+to\s+be\b|\blooks?\s+like\b|"
                r"\bseems?\s+to\s+be\b)"
                rf"\s+(?:an?\s+|the\s+)?{re.escape(term)}\b"
            )
            if re.search(pattern, low):
                substitutions.append({"term": term, "text": text})
                break
    return substitutions


def _collection_substitutions(
    perception_texts: list[str],
    target: str,
) -> list[str]:
    aliases = CLASS_CLAIM_TERMS.get(target, (target,))
    rows: list[str] = []
    for text in perception_texts:
        low = text.lower()
        if any(
            re.search(
                rf"\b(?:multiple|numerous|several|many|rows?\s+of)\s+"
                rf"(?:\w+\s+){{0,2}}{re.escape(alias)}s?\b",
                low,
            )
            for alias in aliases
        ):
            rows.append(text)
    return rows


def audit_trace_semantics(
    item: dict[str, Any],
    raw: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservative semantic-consistency screen before GT teacher forcing."""
    task = item.get("task")
    target = normalize_class_name(item.get("class_name") or str(item.get("gt", "")).split("|", 1)[0])
    loop = raw.get("loop_result") if isinstance(raw.get("loop_result"), dict) else {}
    final = extract_final_text(str(loop.get("final_answer") or ""))
    history = loop.get("chat_history") if isinstance(loop.get("chat_history"), list) else []
    last = _last_evidence_text(loop, 2)
    full_perception = " ".join(str(t.get("P_response") or "") for t in history).lower()
    reasons: list[str] = []

    if not target:
        reasons.append("missing_target_class")
    if re.fullmatch(r"(?:none|no)(?:\|.*)?", final.strip(), re.IGNORECASE):
        reasons.append("explicit_missing_target_final")

    target_text = target or "target"
    negative_patterns = (
        rf"\bno\s+(?:such\s+)?{re.escape(target_text)}\b",
        rf"\bnot\s+(?:an?\s+)?{re.escape(target_text)}\b",
        rf"doesn['’]?t\s+appear\s+to\s+be\s+(?:an?\s+)?{re.escape(target_text)}",
        rf"does\s+not\s+appear\s+to\s+be\s+(?:an?\s+)?{re.escape(target_text)}",
    )
    if any(re.search(pattern, last) for pattern in negative_patterns):
        reasons.append("late_target_denial")

    evidence_terms = CLASS_EVIDENCE.get(target or "", (target or "",))
    explicit_target_term = any(term and term in full_perception for term in evidence_terms)
    perception_texts = [
        str(turn.get("P_response") or "").strip()
        for turn in history
        if str(turn.get("P_response") or "").strip()
    ]
    questions = _reasoner_questions(history)
    threshold = float(
        ((settings or {}).get("trajectory") or {}).get(
            "question_similarity_threshold",
            0.82,
        )
    )
    duplicate_pairs = _duplicate_question_pairs(questions, threshold)
    if duplicate_pairs:
        reasons.append("duplicate_or_highly_similar_question")
    service_error_terms = (
        "apiconnectionerror",
        "connection error",
        "connection refused",
        "service unavailable",
        "traceback",
    )
    positive = any(
        len(text) >= 20 and not any(term in text.lower() for term in service_error_terms)
        for text in perception_texts
    )
    if not positive:
        reasons.append("missing_usable_perceiver_evidence")

    if target in {"large vehicle", "small vehicle"}:
        vehicle_terms = ("vehicle", "truck", "bus", "car", "sedan", "trailer", "van")
        if ("building" in last or "warehouse" in last) and not any(term in last for term in vehicle_terms):
            reasons.append("late_building_vehicle_contradiction")

    leading_questions: list[dict[str, Any]] = []
    denials: list[str] = []
    claims: dict[str, list[str]] = {}
    wrong_claims: dict[str, list[str]] = {}
    scene_substitutions: list[dict[str, str]] = []
    collection_substitutions: list[str] = []
    if task == "ref_grounding_obb":
        width = int(item.get("image_width", 0) or 0)
        height = int(item.get("image_height", 0) or 0)
        coordinate_mode = coordinate_parse_mode(item.get("coordinate_target"))
        has_coordinate_evidence = any(
            bool(
                extract_spatial_candidates(
                    str(turn.get("P_response") or ""),
                    width,
                    height,
                    coordinate_mode,
                    f"perceiver_round_{turn.get('round', '?')}",
                )
            )
            for turn in history
        ) if width > 0 and height > 0 else False
        if not has_coordinate_evidence:
            reasons.append("missing_perceiver_coordinate_evidence")
    elif task == "ref_classification":
        pred = normalize_class_name(final)
        if pred != target:
            reasons.append("classification_final_mismatch")
        leading_questions = _classification_leading_questions(questions)
        if leading_questions:
            reasons.append("classification_answer_obtained_by_class_leading_question")
        denials = _target_denials(perception_texts, target or "")
        claims = _strong_class_claims(perception_texts)
        wrong_claims = {
            class_name: values
            for class_name, values in claims.items()
            if class_name != target
        }
        scene_substitutions = _scene_substitutions(perception_texts, target or "")
        collection_substitutions = _collection_substitutions(
            perception_texts,
            target or "",
        )
        contradiction = bool(
            denials
            or wrong_claims
            or len(claims) > 1
            or scene_substitutions
            or collection_substitutions
        )
        if contradiction:
            reasons.append("classification_visual_evidence_contradiction")
        if scene_substitutions:
            reasons.append("classification_scene_region_replaced_target")
        if collection_substitutions:
            reasons.append("classification_collection_replaced_single_target")
    return {
        "pass": not reasons,
        "task": task,
        "target_class": target,
        "reasons": reasons,
        "positive_visual_evidence": positive,
        "explicit_target_term_in_perception": explicit_target_term,
        "perception_rounds": sum(1 for turn in history if turn.get("P_response")),
        "questions": questions,
        "question_similarity_threshold": threshold,
        "duplicate_question_pairs": duplicate_pairs,
        "duplicate_question_trajectory": bool(duplicate_pairs),
        "classification_leading_questions": leading_questions,
        "classification_target_denials": denials,
        "classification_claimed_classes": sorted(claims),
        "classification_wrong_class_claims": sorted(wrong_claims),
        "classification_scene_substitutions": scene_substitutions,
        "classification_collection_substitutions": collection_substitutions,
        "classification_contradiction": bool(
            task == "ref_classification"
            and (
                denials
                or wrong_claims
                or len(claims) > 1
                or scene_substitutions
                or collection_substitutions
            )
        ),
    }
