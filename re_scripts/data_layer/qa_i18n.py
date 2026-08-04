"""Deterministic Chinese/English rendering for OBB QA.

Only natural-language text changes with ``zh``/``en``.  Structural contracts,
class labels from ``classes.txt``, coordinate modes, GT answers and task IDs
remain deterministic.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

LANGUAGES = {"zh", "en"}

REGION_ZH = {
    "upper-left area": "左上区域",
    "upper-center area": "上方中央区域",
    "upper-right area": "右上区域",
    "middle-left area": "左侧中央区域",
    "central area": "中央区域",
    "middle-right area": "右侧中央区域",
    "lower-left area": "左下区域",
    "lower-center area": "下方中央区域",
    "lower-right area": "右下区域",
}

GEOMETRY_ZH = {
    "strongly elongated": "长轴远长于短轴且具有完整二维目标轮廓",
    "elongated": "长轴明显长于短轴且具有完整二维目标轮廓",
    "compact": "长轴与短轴接近、整体紧凑",
    "moderately rectangular": "中等长方形",
    "roughly horizontal": "大致水平",
    "roughly vertical": "大致垂直",
    "diagonally descending to the right": "向右下方倾斜",
    "diagonally ascending to the right": "向右上方倾斜",
}

GEOMETRY_EN = {
    "strongly elongated": "with a major axis far longer than its minor axis while retaining a complete two-dimensional object outline",
    "elongated": "with a major axis clearly longer than its minor axis while retaining a complete two-dimensional object outline",
    "compact": "with similar major- and minor-axis lengths and a compact outline",
    "moderately rectangular": "with a moderately rectangular outline",
    "roughly horizontal": "oriented roughly horizontally",
    "roughly vertical": "oriented roughly vertically",
    "diagonally descending to the right": "oriented diagonally toward the lower right",
    "diagonally ascending to the right": "oriented diagonally toward the upper right",
}

EXTREME_EN = {
    "leftmost": "leftmost",
    "rightmost": "rightmost",
    "topmost": "highest",
    "bottommost": "lowest",
}
EXTREME_ZH = {
    "leftmost": "最靠左",
    "rightmost": "最靠右",
    "topmost": "最靠上",
    "bottommost": "最靠下",
}

RELATION_ZH = {
    "left of": "位于……左侧",
    "right of": "位于……右侧",
    "above": "位于……上方",
    "below": "位于……下方",
    "above and left of": "位于……左上方",
    "above and right of": "位于……右上方",
    "below and left of": "位于……左下方",
    "below and right of": "位于……右下方",
}


def normalize_lang(value: object) -> str:
    lang = str(value or "en").strip().lower()
    if lang not in LANGUAGES:
        raise ValueError(f"QA language must be zh/en, got {lang!r}")
    return lang


def coordinate_instruction(
    target: str,
    width: int,
    height: int,
    lang: str,
) -> str:
    lang = normalize_lang(lang)
    if lang == "zh":
        if target == "pixel_obb":
            return (
                f"使用原始图像像素坐标，图像尺寸为 {width}×{height}，"
                "左上角为坐标原点"
            )
        if target == "norm100_obb":
            return (
                "使用相对于原始图像的 [0,100] 归一化坐标，"
                "左上角为坐标原点"
            )
        if target == "norm1000_obb":
            return (
                "使用相对于原始图像的 [0,1000] 归一化坐标，"
                "左上角为 (0,0)，右下角为 (1000,1000)"
            )
    else:
        if target == "pixel_obb":
            return (
                f"original-image pixel coordinates for an image of size "
                f"{width}x{height}, with top-left origin"
            )
        if target == "norm100_obb":
            return (
                "normalized coordinates in [0,100] relative to the original "
                "image, with top-left origin"
            )
        if target == "norm1000_obb":
            return (
                "normalized coordinates in [0,1000] relative to the original "
                "image, with top-left origin"
            )
    raise ValueError(f"Unsupported coordinate target: {target}")


def _geometry_zh(phrase: str) -> str:
    parts = [x.strip() for x in str(phrase or "").split(" and ") if x.strip()]
    return "且".join(GEOMETRY_ZH.get(x, x) for x in parts)


def _geometry_en(phrase: str) -> str:
    parts = [x.strip() for x in str(phrase or "").split(" and ") if x.strip()]
    return " and ".join(GEOMETRY_EN.get(x, x) for x in parts)


def _relation_zh(relation: str, anchor_text: str) -> str:
    template = RELATION_ZH.get(relation)
    if template:
        return template.replace("……", anchor_text)
    return f"与{anchor_text}具有“{relation}”关系"


def render_reference(
    *,
    constraints: dict[str, Any],
    class_name: str,
    lang: str,
    masked: bool,
) -> str:
    """Render one reference from language-independent semantic constraints."""
    lang = normalize_lang(lang)
    region = str(constraints.get("region") or "central area")
    geometry = str(constraints.get("geometry_phrase") or "")
    extreme = str(constraints.get("extreme") or "")
    anchor = (
        constraints.get("anchor")
        if isinstance(constraints.get("anchor"), dict)
        else None
    )

    if lang == "zh":
        head = "目标对象" if masked else class_name
        descriptors: list[str] = []
        if geometry:
            descriptors.append(_geometry_zh(geometry))
        descriptors.append(REGION_ZH.get(region, region))
        core = "、".join(descriptors) + f"中的{head}"
        if extreme:
            group = "视觉外观相近的目标" if masked else f"所有“{class_name}”目标"
            core = f"图像中{group}里{EXTREME_ZH.get(extreme, extreme)}的{core}"
        if anchor:
            anchor_text = "另一个视觉上不同的目标对象" if masked else f"“{anchor.get('class_name', '')}”"
            core = f"{core}，{_relation_zh(str(anchor.get('relation') or ''), anchor_text)}"
        return core

    head = "target object" if masked else class_name
    rendered_geometry = _geometry_en(geometry) if geometry else ""
    core = f"the {head} in the {region}"
    if rendered_geometry:
        core = f"{core}, {rendered_geometry}"
    if extreme:
        group = "visually similar targets" if masked else f"all {class_name} instances"
        core = (
            f"the {EXTREME_EN.get(extreme, extreme)} {head} among "
            f"{group} in the image, located in the {region}"
        )
        if rendered_geometry:
            core = f"{core}, {rendered_geometry}"
    if anchor:
        anchor_text = (
            "a visually distinct target object"
            if masked
            else str(anchor.get("class_name") or "anchor target")
        )
        core = f"{core}, {anchor.get('relation')} the {anchor_text}"
    return core


def grounding_question(
    *,
    width: int,
    height: int,
    focus_text: str,
    reference: str,
    coordinate_target: str,
    lang: str,
) -> str:
    instruction = coordinate_instruction(
        coordinate_target, width, height, lang
    )
    if normalize_lang(lang) == "zh":
        return (
            f"图像尺寸：{width}×{height}。目标位于 [0,1000] 归一化坐标下的"
            f"粗略搜索区域 [{focus_text}] 内；该区域只用于提示搜索范围，"
            f"必须根据指代描述确定精确目标。请定位：{reference}。"
            f"{instruction}。只返回一行："
            "class_name|x1,y1,x2,y2,x3,y3,x4,y4，"
            "四个角点按顺时针排列。"
        )
    return (
        f"Image size: {width}x{height}. The target is inside the coarse focus "
        f"region [{focus_text}] in normalized [0,1000] coordinates; this region "
        f"is only a search hint, so resolve the exact target from the reference. "
        f"Locate {reference}. Use {instruction}. Return exactly one line as "
        "class_name|x1,y1,x2,y2,x3,y3,x4,y4 with four corners in clockwise order."
    )


def classification_question(
    *,
    focus_text: str,
    reference: str,
    class_names: Iterable[str],
    lang: str,
) -> str:
    labels = list(class_names)
    if normalize_lang(lang) == "zh":
        choices = "、".join(f"“{x}”" for x in labels)
        return (
            f"目标中心位于 [0,1000] 归一化坐标下的粗略搜索区域 "
            f"[{focus_text}] 内，指代描述为：{reference}。"
            "该区域只用于指向目标，不代表答案。"
            f"请从以下标准类别中选择且只选择一个：{choices}。"
            "必须根据图像证据判断；只返回 classes.txt 中的完整标准类别名。"
        )
    return (
        "The target is centered inside the coarse focus region "
        f"[{focus_text}] in normalized [0,1000] coordinates and is described as "
        f"{reference}. Which canonical category is it? Choose exactly one label "
        f"from: {', '.join(labels)}. Use visual evidence; the focus region is only "
        "an object pointer. Return only the exact canonical class name from "
        "classes.txt."
    )


def direct_detection_question(
    *,
    width: int,
    height: int,
    class_name: str,
    roi_text: str,
    coordinate_target: str,
    lang: str,
) -> str:
    convention = coordinate_instruction(
        coordinate_target, width, height, lang
    )
    if normalize_lang(lang) == "zh":
        return (
            f"图像尺寸：{width}×{height}。检测所有中心点位于 ROI "
            f"[{roi_text}] 内的“{class_name}”目标。使用{convention}。"
            "答案中的坐标始终相对于完整原始图像。"
            "在 FINAL_DETECTIONS 与 END_DETECTIONS 之间，"
            "每个目标输出一行："
            "class_name|confidence|x1,y1,x2,y2,x3,y3,x4,y4，"
            "四个角点按顺时针排列。若不存在匹配目标，返回空区块。"
        )
    return (
        f"Image size: {width}x{height}. Detect every {class_name} object whose "
        f"center lies inside ROI [{roi_text}] using the same coordinate convention. "
        f"Use {convention}. Output one line per object as "
        "class_name|confidence|x1,y1,x2,y2,x3,y3,x4,y4, with corners in clockwise "
        "order, between FINAL_DETECTIONS and END_DETECTIONS. Coordinates in the "
        "answer remain relative to the full original image. If no matching object "
        "exists, return an empty block."
    )



def direct_all_objects_json_question(
    *,
    width: int,
    height: int,
    coordinate_target: str,
    lang: str,
    output_kind: str,
) -> str:
    """Render deterministic full-image all-object Direct QA.

    ``output_kind`` is either ``obb`` or ``hbb``.  Class labels are never
    translated; they remain the exact strings from classes.txt.
    """
    lang = normalize_lang(lang)
    if output_kind not in {"obb", "hbb"}:
        raise ValueError(f"unsupported output_kind: {output_kind}")
    if output_kind == "obb":
        convention = coordinate_instruction(coordinate_target, width, height, lang)
        if lang == "zh":
            return (
                f"图像尺寸：{width}×{height}。请定位图中全部标注目标，不遗漏、不重复。"
                f"使用{convention}。只输出一个合法JSON对象，格式为："
                '{"objects":[{"category":"classes.txt中的完整类别名",'
                '"obb_8":[x1,y1,x2,y2,x3,y3,x4,y4]}]}。'
                "四个角点按顺时针排列；不得输出Markdown、解释或额外文字。"
            )
        return (
            f"Image size: {width}x{height}. Locate every annotated target object in "
            f"the full image without omission or duplication. Use {convention}. "
            "Return only one valid JSON object in this schema: "
            '{"objects":[{"category":"exact label from classes.txt",'
            '"obb_8":[x1,y1,x2,y2,x3,y3,x4,y4]}]}. '
            "Corners must be clockwise. Do not output Markdown or explanation."
        )

    if lang == "zh":
        return (
            f"图像尺寸：{width}×{height}。请定位图中全部标注目标，不遗漏、不重复。"
            "bbox_2d使用相对于完整原图的[0,1000]归一化坐标[x1,y1,x2,y2]。"
            "只输出一个合法JSON对象，格式为："
            '{"bbox_2d":[[x1,y1,x2,y2]],"categories":["classes.txt中的完整类别名"]}。'
            "两个数组必须严格等长且索引一一对应；不得输出Markdown、解释或额外文字。"
        )
    return (
        f"Image size: {width}x{height}. Locate every annotated target object in the "
        "full image without omission or duplication. bbox_2d uses normalized "
        "[0,1000] full-image coordinates [x1,y1,x2,y2]. Return only one valid JSON "
        'object: {"bbox_2d":[[x1,y1,x2,y2]],"categories":["exact label from classes.txt"]}. '
        "The two arrays must have equal lengths and matching indices. No Markdown or explanation."
    )


def direct_single_ref_json_question(
    *,
    width: int,
    height: int,
    focus_text: str,
    reference: str,
    coordinate_target: str,
    lang: str,
) -> str:
    """Render one-target final-only Direct/Ref-style JSON OBB supervision."""
    lang = normalize_lang(lang)
    convention = coordinate_instruction(coordinate_target, width, height, lang)
    if lang == "zh":
        return (
            f"图像尺寸：{width}×{height}。目标位于粗略搜索区域[{focus_text}]内，"
            f"指代描述为：{reference}。粗略区域只用于搜索，不能直接作为目标边界。"
            f"请定位这个单一目标并使用{convention}。只输出一个合法JSON对象："
            '{"category":"classes.txt中的完整类别名",'
            '"obb_8":[x1,y1,x2,y2,x3,y3,x4,y4]}。'
            "四点按顺时针排列；不得逐个询问角点，不得输出解释。"
        )
    return (
        f"Image size: {width}x{height}. The single target is inside coarse focus region "
        f"[{focus_text}] and is described as {reference}. The coarse region is only a "
        f"search hint and must not be copied as the target boundary. Use {convention}. "
        "Return only one valid JSON object: "
        '{"category":"exact label from classes.txt",'
        '"obb_8":[x1,y1,x2,y2,x3,y3,x4,y4]}. '
        "Use clockwise corners; do not output explanation."
    )

def has_focus_phrase(text: str, lang: str) -> bool:
    low = re.sub(r"\s+", " ", str(text or "")).casefold()
    if normalize_lang(lang) == "zh":
        return "粗略搜索区域" in low or "搜索区域" in low
    return "focus region" in low
