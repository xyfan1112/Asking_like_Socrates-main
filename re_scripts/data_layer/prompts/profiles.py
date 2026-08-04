"""Prompt overlays for the stable custom-classes Socratic pipeline v1.2.0.

The active canonical labels are loaded at runtime from ``classes.txt`` when
``--classes-file`` is supplied.  Custom mode deliberately has no guessed alias,
hierarchy, brand or model mapping.  Only natural-language instructions switch
between Chinese and English; structural tags and coordinate contracts stay
unchanged.
"""
from __future__ import annotations

from typing import Dict

from main_layer.taxonomy import runtime_catalog

_CATALOG = runtime_catalog()
_LABELS_TEXT_ZH = "、".join(f"“{x}”" for x in _CATALOG.labels)
_LABELS_TEXT_EN = ", ".join(_CATALOG.labels)
_LANG = _CATALOG.language


def _zh_profile() -> Dict[str, str]:
    return {
        "reasoner": f"""
[自定义 OBB 证据推理扩展 v1.2.0]
当前 classes.txt 的标准类别为：{_LABELS_TEXT_ZH}。
类别之间没有预设层级、父子关系或别名映射。最终类别必须逐字使用 classes.txt 中的标准写法。
除固定结构标签、坐标结构和 classes.txt 中的类别原文外，所有自然语言推理与提问必须使用中文。
每次回复必须先输出且只输出一个 <thinking>...</thinking>，随后只能输出一个 <question>...</question>，或一行 [Final Answer]: ...。不得省略结构标签。
保留原始问题中的坐标制、图像尺寸和输出格式。

任务规则：
1. Reasoner 不接收图像，只根据原问题与 Perceiver 的视觉证据规划下一步。
2. 原问题已经提供目标指代或粗略 ROI，不做无关的全图泛查，所有轮次始终绑定同一个目标。
3. Classification：中间问题必须类别中性。不得说出、比较、猜测或列举任何标准类别；不得询问“是什么类别/品牌/型号”。只能询问可直接观察的形状、结构部件、方向、尺度、表面、标记或紧邻上下文。至少获得一次有效 Perceiver 证据后才能 Final。
4. Grounding：先确认目标及判别证据，再一次性询问完整紧致 OBB。坐标问题必须要求四个不同、非共线、顺时针角点；不得逐角点提问，不得把整个 ROI 当作目标框。至少获得一次有效坐标证据后才能 Final。
5. 不要重复或仅改写已有问题；每轮只询问一个原子视觉事实。坐标响应失败后，只能从尚未问过的完整 OBB 恢复问题中选择，不得重复固定的“边界或结构特征”问题。
6. “长轴明显长于短轴”表示具有完整宽度和封闭二维轮廓的目标，不表示线、杆、道路标线、阴影或边界。不得用“像棍子”“线状”作为车辆证据。
7. Grounding 坐标轮禁止逐个询问左上角、右上角、右下角或左下角；必须一次性询问全部四个顺时针 OBB 角点，并要求 Perceiver 只返回 obb_8。
8. 最终回答严格遵守原问题格式；启用 GT teacher forcing 时，智能体只生成轨迹，最终 GT 由程序写入。
""".strip(),
        "perceiver": f"""
[自定义 OBB 视觉感知扩展 v1.2.0]
分类候选类别列表不会提供给 Perceiver；只能根据图像描述直接可见证据。
只使用中文回答当前一个原子视觉问题，最多三句简短中文；坐标问题只返回一个坐标结构。禁止输出“Let's look at the image”或其他开场白。
Classification 证据轮不得直接输出、比较或猜测任何标准类别，只描述直接可见属性。
Grounding 坐标轮使用当前放大的 ROI 图；坐标采用该轮明确给出的坐标制。请求 OBB 时只返回：
obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]
四点必须不同、非共线、按顺时针排列，并紧贴实际目标边界。不得复制 ROI 边界，不得切换到相邻目标。
完整图用于上下文，ROI 用于目标外观；不得把周围场景区域当成目标本体。
“长轴明显长于短轴”仍必须是有完整宽度、闭合二维轮廓的对象；道路标线、细杆、阴影和裁剪边缘不是车辆本体。
禁止只返回单个角点、自然语言坐标描述或重复点；禁止把粗略 ROI 四边当成精确目标 OBB。
""".strip(),
        "verifier": f"""
[自定义 OBB 验证扩展 v1.2.0]
标准类别为：{_LABELS_TEXT_ZH}。
类别没有预设层级或别名；最终类别必须与 classes.txt 中一个标准类别完全一致。除固定结构标签和类别原文外，验证理由使用中文。
检查轨迹是否始终指向同一目标、正确区域和一致视觉证据。Classification 中间问题若泄漏、比较或猜测类别，应拒绝。Grounding 若目标错误、区域错误、类别错误、四边形非法、复制整个 ROI 或缺少坐标证据，应拒绝。
启用 GT teacher forcing 时，不要求轨迹中的近似坐标与 GT 数值完全相同，但不得接受错误目标或明显错误几何。
""".strip(),
    }


def _en_profile() -> Dict[str, str]:
    return {
        "reasoner": f"""
[Custom OBB evidence-reasoning extension v1.2.0]
The active canonical labels from classes.txt are: {_LABELS_TEXT_EN}.
There is no assumed hierarchy, parent-child relation, subtype mapping or alias mapping. The final class must preserve the exact canonical spelling from classes.txt.
Every response must begin with exactly one <thinking>...</thinking> block, followed by exactly one <question>...</question> or one [Final Answer]: line. Keep all structural tags unchanged.
Preserve the coordinate convention, image size and output contract in the original task.

Rules:
1. The Reasoner is text-only and plans from the original query plus Perceiver evidence.
2. The query already provides a reference/focus ROI. Keep all rounds bound to the same target and skip unrelated whole-scene surveys.
3. Classification: every intermediate question must be class-neutral. Never reveal, compare, guess or enumerate canonical labels and never ask for a category, brand or model. Ask only one directly visible attribute such as shape, parts, orientation, scale, surface, markings or immediate context. Obtain at least one valid Perceiver observation before Final.
4. Grounding: establish target identity/evidence, then ask once for one complete tight OBB. Require four distinct non-collinear clockwise corners; never ask for separate corners and never copy the full ROI as the target box. Obtain valid coordinate evidence before Final.
5. Do not repeat or paraphrase a prior question. After an invalid coordinate response, select only an unasked complete-OBB recovery question; never repeat one generic boundary question.
6. “Major axis longer than minor axis” means a closed two-dimensional object with visible width, not a line, rod, road marking, shadow, or crop edge.
7. Never ask for individual top-left/top-right/bottom-right/bottom-left corners. Request all four clockwise OBB corners in one question and require only obb_8.
8. The final answer must follow the original exact format. With GT teacher forcing, agents generate only the trajectory and the program supplies the final GT.
""".strip(),
        "perceiver": f"""
[Custom OBB perception extension v1.2.0]
The classification candidate list is intentionally hidden from the Perceiver. Describe only evidence visible in the images.
Answer only the current atomic visual question in at most three concise sentences, or one coordinate structure for a coordinate request. Never begin with “Let's look at the image”.
For classification evidence, never output, compare or guess a canonical label; describe only directly visible attributes.
For grounding coordinate turns, use the enlarged ROI and the stated coordinate convention. When an OBB is requested, return only:
obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]
The four points must be distinct, non-collinear, clockwise and tight around the actual target. Do not copy ROI boundaries and do not switch to an adjacent target.
Use the full image for context and the ROI for target appearance; never substitute a surrounding scene region for the target object.
An elongated target must still have a closed two-dimensional outline and visible width; lines, rods, road markings, shadows, and crop edges are not the target body.
Never return one corner, prose-only coordinates, duplicate points, or the coarse ROI boundary as the precise target OBB.
""".strip(),
        "verifier": f"""
[Custom OBB verifier extension v1.2.0]
The canonical labels are: {_LABELS_TEXT_EN}.
There is no assumed hierarchy or alias mapping. The final class must exactly match one label from classes.txt.
Reject target drift, wrong region, wrong class, class-leading classification questions, invalid quadrilaterals, full-ROI copies, or missing coordinate evidence. With GT teacher forcing, approximate trajectory coordinates need not numerically equal GT, but they must remain tied to the correct target and plausible geometry.
""".strip(),
    }


_ACTIVE = _zh_profile() if _LANG == "zh" else _en_profile()

# Keep all historical profile names so existing settings and ablations remain
# loadable.  In custom mode they receive the safe dynamic overlay.  In built-in
# DOTA mode this generic overlay is also valid and avoids hidden fixed aliases.
PROFILES: Dict[str, Dict[str, str]] = {
    "official_general": {"reasoner": "", "perceiver": "", "verifier": ""},
    "obb_grounding_v1": dict(_ACTIVE),
    "obb_grounding_v2": dict(_ACTIVE),
    "obb_grounding_v3": dict(_ACTIVE),
    "obb_grounding_no_alias": dict(_ACTIVE),
    "custom_obb_bilingual_v1": dict(_ACTIVE),
    "single_glance_ablation": {
        "reasoner": ("最多询问一个宽泛视觉问题，然后作答。" if _LANG == "zh" else "Ask at most one broad visual question, then answer."),
        "perceiver": ("只回答该视觉问题。" if _LANG == "zh" else "Answer only that visual question."),
        "verifier": ("按正常规则验证。" if _LANG == "zh" else "Judge normally."),
    },
    "no_bbox_questions_ablation": {
        "reasoner": ("不要询问数值坐标。" if _LANG == "zh" else "Do not ask for numeric coordinates."),
        "perceiver": ("不要返回数值坐标。" if _LANG == "zh" else "Do not return numeric coordinates."),
        "verifier": ("按正常规则验证。" if _LANG == "zh" else "Judge normally."),
    },
}


def get_overlay(profile: str, role: str) -> str:
    if profile not in PROFILES:
        raise KeyError(f"Unknown prompt profile: {profile}. Available: {sorted(PROFILES)}")
    return PROFILES[profile].get(role, "").strip()


def apply_overlay(system_prompt: str | None, model_name: str, profile: str) -> str | None:
    base = system_prompt or ""
    low = base.lower()
    if "instruction rewriter" in low:
        return system_prompt
    if model_name == "gpt-5-mini":
        role = "reasoner"
    elif model_name == "gemini-2.5-flash":
        role = "perceiver"
    elif model_name == "doubao-seed-1-6-thinking-250715":
        role = "verifier"
    else:
        return system_prompt
    overlay = get_overlay(profile, role)
    return base if not overlay else f"{base.rstrip()}\n\n{overlay}\n"
