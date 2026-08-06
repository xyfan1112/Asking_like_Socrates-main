"""Local OpenAI-compatible adapter for the unmodified official SocraticAgent.

re_scripts_ssh1.2.2 complete fix keeps ``SocraticAgent/generation.py`` unchanged and adds
local, observable safeguards:

* preserve structured DOTA queries instead of rewriting ``[0,1000]`` to ``[0,1]``;
* attach the original target context/focus ROI to every Perceiver call;
* restate the coordinate convention to the Perceiver for numeric questions;
* reject repeated/highly-similar Reasoner questions before they enter the trace;
* reject class-leading questions in classification trajectories;
* repair malformed Reasoner and Perceiver outputs with bounded retries;
* isolate ROI-only coordinate turns and map crop coordinates back to the full image;
* fail closed when coordinate evidence remains invalid;
* write finish_reason/token usage logs so true token truncation is measurable.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image

from main_layer.taxonomy import contains_term, multilingual_tokens, runtime_catalog

try:
    from data_layer.prompts.profiles import apply_overlay
except Exception:
    apply_overlay = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


MAX_RETRIES = int(os.getenv("LOCAL_AGENT_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = float(os.getenv("LOCAL_AGENT_TIMEOUT", "180"))
MAX_REPAIR_ATTEMPTS = max(0, int(os.getenv("ALS_MAX_REPAIR_ATTEMPTS", "3")))
MAX_LOOP = max(1, int(os.getenv("ALS_MAX_LOOP", "8")))
FORCE_FINAL_ON_LAST_ROUND = _env_bool("ALS_FORCE_FINAL_ON_LAST_ROUND", True)
BYPASS_STRUCTURED_REWRITE = _env_bool("ALS_BYPASS_STRUCTURED_REWRITE", True)
COORDINATE_TARGET = (os.getenv("ALS_COORDINATE_TARGET", "norm1000_obb") or "norm1000_obb").strip()
API_LOG_PATH = (os.getenv("ALS_API_LOG_PATH", "") or "").strip()
QUESTION_SIMILARITY_THRESHOLD = float(
    os.getenv("ALS_QUESTION_SIMILARITY_THRESHOLD", "0.95")
)
REQUIRE_PERCEIVER_CONTEXT = _env_bool("ALS_REQUIRE_PERCEIVER_CONTEXT", True)
ENABLE_FOCUS_CROP = _env_bool("ALS_ENABLE_FOCUS_CROP", True)
FOCUS_CROP_LONG_SIDE = max(256, int(os.getenv("ALS_FOCUS_CROP_LONG_SIDE", "1024")))
FOCUS_CROP_PADDING_RATIO = max(0.0, float(os.getenv("ALS_FOCUS_CROP_PADDING_RATIO", "0.02")))
MAP_CROP_COORDINATES = _env_bool("ALS_MAP_CROP_COORDINATES", True)
DETERMINISTIC_STRUCTURED_VERIFIER = _env_bool("ALS_DETERMINISTIC_STRUCTURED_VERIFIER", True)
VERIFY_MIN_HBB_IOU = float(os.getenv("ALS_VERIFY_MIN_HBB_IOU", "0.15"))
VERIFY_EXPANDED_GT_RATIO = float(os.getenv("ALS_VERIFY_EXPANDED_GT_RATIO", "0.35"))
REASONER_COMPACT_MAX_CHARS = max(3000, int(os.getenv("ALS_REASONER_COMPACT_MAX_CHARS", "9000")))
CLASSIFICATION_USE_FULL_AND_CROP = _env_bool(
    "ALS_CLASSIFICATION_USE_FULL_AND_CROP", True
)
CLASSIFICATION_CROP_ONLY_MAX_AREA = min(1.0, max(0.0, float(
    os.getenv("ALS_CLASSIFICATION_CROP_ONLY_MAX_AREA", "0.00")
)))
CLASSIFICATION_REJECT_CLASS_CLAIMS = _env_bool(
    "ALS_CLASSIFICATION_REJECT_CLASS_CLAIMS", False
)
GROUNDING_COORDINATE_CROP_ONLY = _env_bool(
    "ALS_GROUNDING_COORDINATE_CROP_ONLY", True
)
REJECT_UNSOLICITED_COORDINATES = _env_bool(
    "ALS_REJECT_UNSOLICITED_COORDINATES", True
)
PERCEIVER_FAIL_CLOSED = _env_bool("ALS_PERCEIVER_FAIL_CLOSED", True)
REASONER_FAIL_CLOSED = _env_bool("ALS_REASONER_FAIL_CLOSED", True)
MAX_CROP_COORDINATE_COVERAGE = min(1.0, max(0.5, float(
    os.getenv("ALS_MAX_CROP_COORDINATE_COVERAGE", "0.94")
)))
# Qwen3-Omni defaults to text+audio output in vLLM-Omni when modalities are
# omitted.  Trajectory generation needs text only.  Keep this opt-in so the
# normal Qwen3-VL route is byte-for-byte unchanged.
OMNI_TEXT_ONLY = _env_bool("ALS_OMNI_TEXT_ONLY", False)

_RUNTIME_CATALOG = runtime_catalog()
_QA_LANG = _RUNTIME_CATALOG.language
_CUSTOM_TAXONOMY = _RUNTIME_CATALOG.custom

MODEL_ROUTE: Dict[str, Tuple[str, str]] = {
    "gpt-5-mini": (
        os.getenv("LOCAL_REASONER_BASE_URL", "http://127.0.0.1:8001/v1"),
        os.getenv("LOCAL_REASONER_MODEL", "local-reasoner"),
    ),
    "gemini-2.5-flash": (
        os.getenv("LOCAL_PERCEIVER_BASE_URL", "http://127.0.0.1:8002/v1"),
        os.getenv("LOCAL_PERCEIVER_MODEL", "local-perceiver"),
    ),
    "doubao-seed-1-6-thinking-250715": (
        os.getenv("LOCAL_VERIFIER_BASE_URL", "http://127.0.0.1:8003/v1"),
        os.getenv("LOCAL_VERIFIER_MODEL", "local-verifier"),
    ),
}

_LOG_LOCK = threading.Lock()
_THREAD_CONTEXT = threading.local()
_THINK_RE = re.compile(r"<thinking>\s*([\s\S]*?)\s*</thinking>", re.IGNORECASE)
_QUESTION_RE = re.compile(r"<question>\s*([\s\S]*?)\s*</question>", re.IGNORECASE)
_FINAL_RE = re.compile(r"\[Final Answer\]\s*:\s*(.+)", re.IGNORECASE)
_COORD_TERMS_RE = re.compile(
    r"(?:"
    r"\b(?:coordinate|coordinates|bbox|bounding box|box_2d|hbb|obb|corner|corners|"
    r"polygon|position|location|extent|envelope|locali[sz])\b"
    r"|坐标|角点|四角点|四个角点|顺时针|旋转框|边界框|外接矩形|定位框|包围框"
    r")",
    re.IGNORECASE,
)
_NUM_PATTERN = r"-?\d+(?:\.\d+)?"
_ROI_PREFIX = (
    r"(?:focus\s+region|roi|粗略搜索区域|搜索区域|目标区域|候选区域|"
    r"坐标范围|区域|范围)"
)
_FOCUS_ROI_RE = re.compile(
    rf"{_ROI_PREFIX}[^\[\n]{{0,160}}\[\s*-?\d+(?:\.\d+)?"
    rf"(?:\s*,\s*-?\d+(?:\.\d+)?){3}\s*\]",
    re.IGNORECASE,
)
_FOCUS_ROI_CAPTURE_RE = re.compile(
    rf"{_ROI_PREFIX}[^\[\n]{{0,160}}"
    rf"\[\s*({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*\]",
    re.IGNORECASE,
)
_QUESTION_ROI_RE = re.compile(
    rf"(?:region|roi|bbox|box|coordinates?|粗略搜索区域|搜索区域|目标区域|"
    rf"候选区域|区域|范围|坐标)[^\[\n]{{0,160}}"
    rf"\[\s*({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*\]",
    re.IGNORECASE,
)
_STRICT_BBOX_RESPONSE_RE = re.compile(
    rf"\s*bbox_2d\s*=\s*\[\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*\]\s*",
    re.IGNORECASE,
)
_STRICT_OBB_RESPONSE_RE = re.compile(
    rf"\s*obb_8\s*=\s*\[\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*,\s*"
    rf"({_NUM_PATTERN})\s*\]\s*",
    re.IGNORECASE,
)
_PIPE_OBB_RESPONSE_RE = re.compile(
    rf"[^\n|]{{1,200}}\|\s*({_NUM_PATTERN})"
    rf"(?:\s*,\s*{_NUM_PATTERN}){{7}}",
    re.IGNORECASE,
)
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
_CLASS_LEADING_TERMS = (
    tuple(_RUNTIME_CATALOG.labels)
    if _CUSTOM_TAXONOMY
    else (
        "plane", "aircraft", "airplane", "ship", "boat", "vessel",
        "storage tank", "baseball diamond", "baseball field",
        "tennis court", "basketball court", "ground track field",
        "running track", "harbor", "marina", "bridge",
        "large vehicle", "truck", "bus", "small vehicle", "car",
        "helicopter", "roundabout", "soccer ball field", "soccer field",
        "football field", "swimming pool",
    )
)


def _canonicalize_reasoner_response(text: str) -> tuple[str, bool]:
    """Normalize harmless complete tag variants without inventing content."""
    original = str(text or "").strip()
    out = original.replace("```xml", "").replace("```", "").strip()
    out = re.sub(r"<\s*think\s*>", "<thinking>", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*/\s*think\s*>", "</thinking>", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*ques(?:tion)?\s*>", "<question>", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*/\s*ques(?:tion)?\s*>", "</question>", out, flags=re.IGNORECASE)
    out = re.sub(
        r"\[\s*Question\s*\]\s*:\s*([\s\S]+)$",
        lambda match: f"<question>{match.group(1).strip()}</question>",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"(?m)^\s*Final\s+Answer\s*:\s*",
        "[Final Answer]: ",
        out,
        flags=re.IGNORECASE,
    )
    if _THINK_RE.search(out) is None:
        boundary = re.search(r"<question>|\[Final Answer\]\s*:", out, re.IGNORECASE)
        if boundary and out[: boundary.start()].strip():
            prefix = out[: boundary.start()].strip()
            suffix = out[boundary.start():].strip()
            out = f"<thinking>{prefix}</thinking>\n{suffix}"
        elif boundary:
            out = (
                "<thinking>Format-normalized response based on the collected "
                "visual evidence.</thinking>\n" + out
            )
    return out, out != original


def _looks_incomplete_response(text: str) -> bool:
    """Detect visibly cut-off model text even when finish_reason is not length."""
    raw = str(text or "").strip()
    if not raw:
        return True
    low = raw.casefold().rstrip()
    suspicious_suffixes = (
        "let’s look at the im",
        "let's look at the im",
        "</que",
        "</ques",
        "<question",
        "</thin",
        "</think",
        "<thinking",
    )
    if low.endswith(suspicious_suffixes):
        return True
    # An opening structural tag without its closing tag is not a complete response.
    if re.search(r"<thinking>", raw, re.IGNORECASE) and not re.search(
        r"</thinking>", raw, re.IGNORECASE
    ):
        return True
    if re.search(r"<question>", raw, re.IGNORECASE) and not re.search(
        r"</question>", raw, re.IGNORECASE
    ):
        return True
    return False


def _write_log(record: Dict[str, Any]) -> None:
    if not API_LOG_PATH:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "thread": threading.get_ident(),
        "qa_language": _QA_LANG,
        "taxonomy_mode": "custom" if _CUSTOM_TAXONOMY else "dota",
        "taxonomy_sha256": _RUNTIME_CATALOG.sha256,
        **record,
    }
    path = Path(API_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _encode_image(image: Image.Image, suffix: str) -> str:
    fmt = (suffix or "jpeg").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt == "JPEG" and image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _sampling(model_name: str) -> Tuple[int, float, float]:
    if model_name == "gpt-5-mini":
        return (
            int(os.getenv("LOCAL_REASONER_MAX_TOKENS", "896")),
            float(os.getenv("LOCAL_REASONER_TEMPERATURE", "0.1")),
            float(os.getenv("LOCAL_REASONER_TOP_P", "0.9")),
        )
    if model_name == "gemini-2.5-flash":
        return (
            int(os.getenv("LOCAL_PERCEIVER_MAX_TOKENS", "512")),
            float(os.getenv("LOCAL_PERCEIVER_TEMPERATURE", "0.0")),
            float(os.getenv("LOCAL_PERCEIVER_TOP_P", "0.9")),
        )
    return (
        int(os.getenv("LOCAL_VERIFIER_MAX_TOKENS", "256")),
        float(os.getenv("LOCAL_VERIFIER_TEMPERATURE", "0.0")),
        float(os.getenv("LOCAL_VERIFIER_TOP_P", "0.9")),
    )


def _extract_original_query(query: str) -> str:
    marker = "Original query:"
    if marker in query:
        return query.split(marker, 1)[1].strip()
    return query.strip()


def _extract_reasoner_target_context(query: str) -> str:
    """Extract only the immutable user target and image metadata.

    The official generator appends prior Reasoner/Perceiver turns after the
    metadata.  Keeping only the pre-history block prevents the context injected
    into Perceiver calls from growing every round.
    """
    text = str(query or "")
    user_marker = "# User Query:"
    meta_marker = "# Image Metadata:"
    if user_marker not in text:
        return ""
    after_user = text.split(user_marker, 1)[1]
    if meta_marker in after_user:
        user_query, after_meta = after_user.split(meta_marker, 1)
        metadata = re.split(r"<thinking>|<question>", after_meta, maxsplit=1, flags=re.IGNORECASE)[0]
    else:
        user_query, metadata = after_user, ""
    user_query = user_query.strip()
    metadata = metadata.strip()
    if not user_query:
        return ""
    parts = [user_query]
    if metadata:
        parts.append(f"Image metadata:\n{metadata}")
    return "\n\n".join(parts)


def _reset_perceiver_evidence() -> None:
    _THREAD_CONTEXT.valid_perceiver_calls = 0
    _THREAD_CONTEXT.valid_coordinate_calls = 0
    _THREAD_CONTEXT.valid_semantic_calls = 0


def _record_perceiver_evidence(coordinate_kind: str) -> None:
    _THREAD_CONTEXT.valid_perceiver_calls = int(
        getattr(_THREAD_CONTEXT, "valid_perceiver_calls", 0)
    ) + 1
    if coordinate_kind == "none":
        _THREAD_CONTEXT.valid_semantic_calls = int(
            getattr(_THREAD_CONTEXT, "valid_semantic_calls", 0)
        ) + 1
    else:
        _THREAD_CONTEXT.valid_coordinate_calls = int(
            getattr(_THREAD_CONTEXT, "valid_coordinate_calls", 0)
        ) + 1


def _current_evidence_counts() -> Tuple[int, int, int]:
    return (
        int(getattr(_THREAD_CONTEXT, "valid_perceiver_calls", 0)),
        int(getattr(_THREAD_CONTEXT, "valid_coordinate_calls", 0)),
        int(getattr(_THREAD_CONTEXT, "valid_semantic_calls", 0)),
    )


def _remember_reasoner_context(query: str) -> str:
    context = _extract_reasoner_target_context(query)
    if not context:
        _THREAD_CONTEXT.original_target_context = ""
        _THREAD_CONTEXT.original_target_sha256 = ""
        _THREAD_CONTEXT.original_target_task = ""
        _reset_perceiver_evidence()
        return ""

    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
    previous_digest = str(
        getattr(_THREAD_CONTEXT, "original_target_sha256", "") or ""
    )
    if digest != previous_digest:
        _reset_perceiver_evidence()

    _THREAD_CONTEXT.original_target_context = context
    _THREAD_CONTEXT.original_target_sha256 = digest
    _THREAD_CONTEXT.original_target_task = (
        "ref_classification"
        if _is_classification_context(context)
        else "ref_grounding_obb"
    )
    return context


def _current_reasoner_context() -> str:
    return str(getattr(_THREAD_CONTEXT, "original_target_context", "") or "")


def _is_classification_context(context: str) -> bool:
    text = str(context or "")
    low = re.sub(r"\s+", " ", text.casefold())
    markers = (
        "[task=ref_classification]",
        "which canonical dota category",
        "which canonical category",
        "return only the canonical class name",
        "return only the exact canonical class name",
        "choose exactly one label from",
        "请从以下标准类别中选择且只选择一个",
        "只返回 classes.txt 中的完整标准类别名",
        "只返回classes.txt中的完整标准类别名",
        "只返回标准类别名",
    )
    return any(marker in low for marker in markers)


def _is_grounding_context(context: str) -> bool:
    text = str(context or "")
    low = re.sub(r"\s+", " ", text.casefold())
    return (
        "[task=ref_grounding_obb]" in low
        or "class_name|x1,y1,x2,y2,x3,y3,x4,y4" in low
        or "只返回一行：class_name|x1,y1,x2,y2,x3,y3,x4,y4" in text
    )


def _normalized_question_tokens(question: str) -> list[str]:
    return [
        token for token in multilingual_tokens(question)
        if token not in _QUESTION_STOPWORDS
    ]


def question_similarity(left: str, right: str) -> float:
    """Return a conservative lexical/sequence similarity in [0, 1]."""
    left_tokens = _normalized_question_tokens(left)
    right_tokens = _normalized_question_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_norm = " ".join(left_tokens)
    right_norm = " ".join(right_tokens)
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_set, right_set = set(left_tokens), set(right_tokens)
    jaccard = len(left_set & right_set) / max(1, len(left_set | right_set))
    return max(float(sequence), float(jaccard))


def _duplicate_question(
    question: str,
    previous_questions: List[str],
) -> tuple[bool, float, Optional[str]]:
    best_score = 0.0
    best_previous: Optional[str] = None
    normalized = " ".join(_normalized_question_tokens(question))
    for previous in previous_questions:
        previous_normalized = " ".join(_normalized_question_tokens(previous))
        score = 1.0 if normalized and normalized == previous_normalized else question_similarity(
            question, previous
        )
        if score > best_score:
            best_score, best_previous = score, previous
    return (
        best_score >= QUESTION_SIMILARITY_THRESHOLD,
        best_score,
        best_previous,
    )


def _classification_question_is_leading(question: str) -> bool:
    low = re.sub(r"\s+", " ", str(question or "").casefold()).strip()
    if re.search(
        r"\b(?:canonical\s+dota\s+)?(?:category|class|label|brand|model)\b",
        low,
    ) or any(term in low for term in ("类别", "分类", "标签", "品牌", "型号")):
        return True
    return any(contains_term(low, term) for term in _CLASS_LEADING_TERMS)


def _extract_focus_roi(context: str) -> Optional[Tuple[float, float, float, float]]:
    match = _FOCUS_ROI_CAPTURE_RE.search(str(context or ""))
    if not match:
        return None
    values = tuple(float(value) for value in match.groups())
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return values


def _polygon_signed_area(points: List[Tuple[float, float]]) -> float:
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(
    a: Tuple[float, float], b: Tuple[float, float],
    c: Tuple[float, float], d: Tuple[float, float],
) -> bool:
    eps = 1e-6
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def _valid_obb_values(values: List[float]) -> Tuple[bool, str]:
    if len(values) != 8 or any(not math.isfinite(v) for v in values):
        return False, "invalid_obb_number_count"
    if any(v < 0 or v > 1000 for v in values):
        return False, "coordinate_out_of_range"
    points = [(values[i], values[i + 1]) for i in range(0, 8, 2)]
    rounded = {(round(x, 3), round(y, 3)) for x, y in points}
    if len(rounded) < 4:
        return False, "obb_duplicate_points"
    if abs(_polygon_signed_area(points)) < 4.0:
        return False, "obb_zero_or_tiny_area"
    if _segments_cross(points[0], points[1], points[2], points[3]) or _segments_cross(
        points[1], points[2], points[3], points[0]
    ):
        return False, "obb_self_intersection"
    return True, "ok"


def _order_obb_clockwise(values: List[float]) -> List[float]:
    """Order four distinct corner points clockwise and start near top-left.

    Qwen occasionally returns the correct four corners in a crossing order.  We
    normalize ordering before validating the polygon; duplicate or collinear
    points are still rejected.
    """
    if len(values) != 8:
        return values
    points = [(float(values[i]), float(values[i + 1])) for i in range(0, 8, 2)]
    if len({(round(x, 6), round(y, 6)) for x, y in points}) < 4:
        return values
    cx = sum(x for x, _ in points) / 4.0
    cy = sum(y for _, y in points) / 4.0
    # In image coordinates y grows downward, increasing atan2 angle follows a
    # clockwise visual traversal.
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(4), key=lambda i: (ordered[i][1], ordered[i][0]))
    ordered = ordered[start:] + ordered[:start]
    flattened: List[float] = []
    for x, y in ordered:
        flattened.extend([x, y])
    return flattened


def _crop_coordinate_coverage(values: List[float]) -> float:
    if len(values) != 8:
        return 0.0
    xs, ys = values[0::2], values[1::2]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys)) / 1_000_000.0


def _build_focus_crop(
    images: Optional[List[Tuple[Image.Image, str]]],
    original_context: str,
) -> Tuple[Optional[List[Tuple[Image.Image, str]]], Optional[Dict[str, Any]]]:
    """Attach a real enlarged ROI as Image 2 while keeping Image 1 as the full scene."""
    if not ENABLE_FOCUS_CROP or not images:
        return images, None
    focus = _extract_focus_roi(original_context)
    if focus is None:
        return images, None
    full = images[0][0]
    width, height = full.size
    if width <= 1 or height <= 1:
        return images, None
    fx1, fy1, fx2, fy2 = focus
    x1, y1 = fx1 * width / 1000.0, fy1 * height / 1000.0
    x2, y2 = fx2 * width / 1000.0, fy2 * height / 1000.0
    pad_x = max(2.0, (x2 - x1) * FOCUS_CROP_PADDING_RATIO)
    pad_y = max(2.0, (y2 - y1) * FOCUS_CROP_PADDING_RATIO)
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right - left < 2 or bottom - top < 2:
        return images, None
    crop = full.crop((left, top, right, bottom)).convert("RGB")
    cw, ch = crop.size
    scale = FOCUS_CROP_LONG_SIDE / max(cw, ch)
    if scale > 1.0:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        crop = crop.resize(
            (max(2, int(round(cw * scale))), max(2, int(round(ch * scale)))),
            resampling,
        )
    crop_norm = [
        left * 1000.0 / width,
        top * 1000.0 / height,
        right * 1000.0 / width,
        bottom * 1000.0 / height,
    ]
    meta = {
        "pixel_bounds": [left, top, right, bottom],
        "norm1000_bounds": crop_norm,
        "original_size": [width, height],
        "crop_size": list(crop.size),
    }
    return list(images) + [(crop, "png")], meta


def _map_crop_values_to_original(values: List[float], crop_meta: Dict[str, Any]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in crop_meta["norm1000_bounds"]]
    mapped: List[float] = []
    for index, value in enumerate(values):
        if index % 2 == 0:
            mapped.append(x1 + value * (x2 - x1) / 1000.0)
        else:
            mapped.append(y1 + value * (y2 - y1) / 1000.0)
    return mapped


def _canonicalize_perceiver_coordinate_response(
    response: str,
    kind: str,
    crop_meta: Optional[Dict[str, Any]],
) -> Tuple[str, bool, str]:
    """Extract, normalize, map and validate one coordinate structure."""
    if kind == "none":
        return str(response or "").strip(), False, "ok"
    text = str(response or "")
    bbox_matches = list(_STRICT_BBOX_RESPONSE_RE.finditer(text))
    obb_matches = list(_STRICT_OBB_RESPONSE_RE.finditer(text))
    matches: List[Tuple[str, Any]] = []
    if kind in {"bbox", "either"}:
        matches.extend(("bbox", m) for m in bbox_matches)
    if kind in {"obb", "either"}:
        matches.extend(("obb", m) for m in obb_matches)
    if len(matches) != 1:
        return text.strip(), False, "coordinate_response_requires_exactly_one_structure"

    detected_kind, match = matches[0]
    local_values = [float(v) for v in match.groups()]
    if any(not math.isfinite(v) or v < 0 or v > 1000 for v in local_values):
        return text.strip(), False, "coordinate_out_of_range"

    if detected_kind == "obb":
        local_values = _order_obb_clockwise(local_values)
        local_valid, local_reason = _valid_obb_values(local_values)
        if not local_valid:
            return text.strip(), False, local_reason
        local_coverage = _crop_coordinate_coverage(local_values)
    else:
        local_coverage = (
            max(0.0, local_values[2] - local_values[0])
            * max(0.0, local_values[3] - local_values[1])
            / 1_000_000.0
        )
    if crop_meta is not None and local_coverage >= MAX_CROP_COORDINATE_COVERAGE:
        return text.strip(), False, "coordinate_copies_entire_crop"

    values = list(local_values)
    if MAP_CROP_COORDINATES and crop_meta is not None:
        values = _map_crop_values_to_original(values, crop_meta)
    values = [max(0, min(1000, int(round(v)))) for v in values]

    if detected_kind == "bbox":
        if values[2] <= values[0] or values[3] <= values[1]:
            return text.strip(), False, "bbox_degenerate"
        canonical = f"bbox_2d=[{','.join(map(str, values))}]"
    else:
        values = [int(round(v)) for v in _order_obb_clockwise([float(v) for v in values])]
        valid, reason = _valid_obb_values([float(v) for v in values])
        if not valid:
            return text.strip(), False, reason
        canonical = f"obb_8=[{','.join(map(str, values))}]"
    return canonical, canonical != text.strip(), "ok"


def _question_rois(question: str) -> List[Tuple[float, float, float, float]]:
    rois: List[Tuple[float, float, float, float]] = []
    for match in _QUESTION_ROI_RE.finditer(str(question or "")):
        values = tuple(float(value) for value in match.groups())
        x1, y1, x2, y2 = values
        if x2 > x1 and y2 > y1:
            rois.append(values)
    return rois


def _point_in_expanded_hbb(
    point: Tuple[float, float],
    hbb: Tuple[float, float, float, float],
    ratio: float = 0.35,
) -> bool:
    x, y = point
    x1, y1, x2, y2 = hbb
    margin_x = max(1.0, (x2 - x1) * ratio)
    margin_y = max(1.0, (y2 - y1) * ratio)
    return (
        x1 - margin_x <= x <= x2 + margin_x
        and y1 - margin_y <= y <= y2 + margin_y
    )


def _roi_is_related(
    candidate: Tuple[float, float, float, float],
    focus: Tuple[float, float, float, float],
) -> bool:
    cx1, cy1, cx2, cy2 = candidate
    fx1, fy1, fx2, fy2 = focus
    intersection_w = max(0.0, min(cx2, fx2) - max(cx1, fx1))
    intersection_h = max(0.0, min(cy2, fy2) - max(cy1, fy1))
    if intersection_w * intersection_h > 0.0:
        return True
    candidate_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
    focus_center = ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0)
    return _point_in_expanded_hbb(
        candidate_center, focus
    ) or _point_in_expanded_hbb(focus_center, candidate)


def _unrelated_question_roi(
    question: str,
    original_context: str,
) -> Optional[Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]]:
    focus = _extract_focus_roi(original_context)
    if focus is None:
        return None
    for candidate in _question_rois(question):
        if not _roi_is_related(candidate, focus):
            return candidate, focus
    return None


def _coordinate_answer_kind(question: str) -> str:
    """Return ``bbox``, ``obb``, ``either`` or ``none`` for a bilingual question."""
    low = re.sub(r"\s+", " ", str(question or "").casefold())
    if not _COORD_TERMS_RE.search(low):
        return "none"
    if re.search(
        r"(?:\b(?:obb(?:_8)?|rotated\s+(?:box|bounding\s+box)|polygon|"
        r"four\s+corners?|clockwise)\b|旋转框|四个角点|四角点|顺时针|"
        r"obb坐标|obb角点)",
        low,
    ):
        return "obb"
    if re.search(
        r"(?:\b(?:bbox(?:_2d)?|box_2d|bounding\s+box|coarse\s+(?:box|hbb)|"
        r"axis[- ]aligned|envelope|top[- ]left.*bottom[- ]right)\b|"
        r"水平框|外接矩形|左上角.*右下角|粗略框)",
        low,
    ):
        return "bbox"
    return "either"


def _coordinate_contract(kind: str) -> str:
    if kind == "bbox":
        return (
            "Return exactly one line and nothing else: "
            "bbox_2d=[xmin,ymin,xmax,ymax]. The box must tightly bound the exact "
            "target, not the full crop or coarse search region."
        )
    if kind == "obb":
        return (
            "Return exactly one line and nothing else: "
            "obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]. Use four distinct, non-collinear "
            "corners in clockwise order around a tight target boundary. Never repeat "
            "a point, never return crossing diagonals, and never copy the full crop "
            "rectangle or the coarse search region."
        )
    if kind == "either":
        return (
            "Return exactly one line and nothing else, using either "
            "bbox_2d=[xmin,ymin,xmax,ymax] or "
            "obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]. The geometry must tightly bound "
            "the exact target and must not copy the full crop."
        )
    return ""


def _perceiver_coordinate_status(response: str, kind: str) -> Tuple[bool, str]:
    if kind == "none":
        return True, "ok"
    bbox = _STRICT_BBOX_RESPONSE_RE.fullmatch(str(response or ""))
    obb = _STRICT_OBB_RESPONSE_RE.fullmatch(str(response or ""))
    if kind == "bbox" and bbox is None:
        return False, "coordinate_response_requires_exact_bbox_2d"
    if kind == "obb" and obb is None:
        return False, "coordinate_response_requires_exact_obb_8"
    if kind == "either" and bbox is None and obb is None:
        return False, "coordinate_response_requires_bbox_2d_or_obb_8"
    return True, "ok"


def _classification_response_reveals_class(response: str) -> bool:
    low = re.sub(r"\s+", " ", str(response or "").lower()).strip()
    if re.search(r"\b(?:canonical\s+dota\s+)?(?:category|class|label)\b", low):
        return True
    return any(
        re.search(rf"\b{re.escape(term)}s?\b", low)
        for term in _CLASS_LEADING_TERMS
    )


def _contains_unsolicited_coordinate_payload(response: str) -> bool:
    text = str(response or "")
    return bool(
        _STRICT_BBOX_RESPONSE_RE.search(text)
        or _STRICT_OBB_RESPONSE_RE.search(text)
        or _PIPE_OBB_RESPONSE_RE.search(text)
    )


def _perceiver_response_status(
    response: str,
    atomic_question: str,
    original_context: str,
) -> Tuple[bool, str]:
    classification_task = _is_classification_context(original_context)
    coordinate_kind = (
        "none" if classification_task else _coordinate_answer_kind(atomic_question)
    )
    text = re.sub(r"\s+", " ", str(response or "")).strip()
    if not text:
        return False, "empty_perceiver_response"
    if _looks_incomplete_response(response):
        return False, "incomplete_perceiver_response"
    if any(term in text.lower() for term in (
        "apiconnectionerror", "connection refused", "service unavailable", "traceback"
    )):
        return False, "perceiver_service_error_text"
    if (
        REJECT_UNSOLICITED_COORDINATES
        and coordinate_kind == "none"
        and _contains_unsolicited_coordinate_payload(response)
    ):
        return False, "unsolicited_coordinate_payload"
    if classification_task:
        # The prompt still asks for class-neutral evidence, but Qwen3-VL often
        # prefixes a useful visual description with a class noun (for example
        # "the plane has wings..."). Hard-rejecting that entire response caused
        # repair loops and removed otherwise valid evidence in ssh1.2.2.
        if (
            CLASSIFICATION_REJECT_CLASS_CLAIMS
            and _classification_response_reveals_class(response)
        ):
            return False, "classification_perceiver_reveals_class_or_alias"
        evidence_units = multilingual_tokens(text)
        if len(evidence_units) < 4:
            return False, "classification_evidence_too_short"
        return True, "ok"
    return _perceiver_coordinate_status(response, coordinate_kind)


def _is_structured_task(query: str) -> bool:
    low = query.lower()
    markers = (
        "[0,1000]",
        "[0, 1000]",
        "[0,100]",
        "[0, 100]",
        "class_name|",
        "x1,y1,x2,y2",
        "four corners",
        "clockwise order",
        "original-image pixel",
        "dota category",
        "canonical dota",
        "return exactly one line",
        "return only one class name",
        "只返回一行",
        "四个角点",
        "顺时针",
        "完整标准类别名",
        "标准类别",
    )
    return any(marker in low for marker in markers)


def _infer_reasoner_round(query: str) -> int:
    # Official history concatenates every previous valid Reasoner question.
    return len(_QUESTION_RE.findall(query)) + 1


def _has_pathological_repetition(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    sentences = [part.strip(" .,:;-") for part in re.split(r"[.!?\n]+", normalized)]
    counts: Dict[str, int] = {}
    for sentence in sentences:
        if len(sentence.split()) < 7:
            continue
        counts[sentence] = counts.get(sentence, 0) + 1
        if counts[sentence] >= 3:
            return True
    return False


def _coordinate_question_is_unsafe(question: str) -> Optional[str]:
    low = re.sub(r"\s+", " ", str(question or "").casefold())
    asks_single_corner = bool(re.search(
        r"(?:\b(?:top[- ]left|top[- ]right|bottom[- ]left|bottom[- ]right) corner\b|"
        r"左上角|右上角|右下角|左下角)",
        low,
    ))
    asks_complete_obb = bool(re.search(
        r"(?:four corners|all four|complete.*(?:obb|box)|四个角点|四角点|完整.*(?:obb|旋转框)|顺时针.*角点)",
        low,
    ))
    if asks_single_corner and not asks_complete_obb:
        return "partial_corner_coordinate_question"
    if re.search(r"(?:\bare (?:the )?(?:four )?corners?.*\d|这些角点是否|坐标是否为|是否是.*\d)", low):
        return "proposed_coordinates_for_confirmation"
    if re.search(r"(?:orientation angle.*degrees|方向角.*度|倾斜角.*度)", low):
        return "numeric_angle_instead_of_direct_visual_geometry"
    return None


def _compact_reasoner_query(query: str, original_context: str, max_chars: int) -> str:
    compact = _THINK_RE.sub("<thinking>[prior reasoning omitted]</thinking>", str(query or ""))
    compact = re.sub(r"(?:Re-evaluating|Rechecking):[^\n]{40,}(?:\n|$)", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
    if len(compact) <= max_chars:
        return compact
    context = str(original_context or "").strip()
    reserve = max(1200, max_chars - len(context) - 160)
    tail = compact[-reserve:]
    return (
        "# Immutable target context\n" + context +
        "\n\n# Recent accepted evidence (old reasoning omitted)\n" + tail
    )[-max_chars:]


def _canonical_class_name(value: str) -> Optional[str]:
    return _RUNTIME_CATALOG.canonicalize(value)



def _parse_structured_answer(value: str) -> Tuple[Optional[str], Optional[List[float]]]:
    text = str(value or "").strip()
    if "|" not in text:
        return _canonical_class_name(text), None
    class_part, coords = text.split("|", 1)
    numbers = [float(v) for v in re.findall(_NUM_PATTERN, coords)]
    return _canonical_class_name(class_part), numbers if len(numbers) == 8 else None


def _hbb_from_values(values: List[float]) -> Tuple[float, float, float, float]:
    xs, ys = values[0::2], values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _hbb_iou_norm(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(1e-9, aa + bb - inter)


def _deterministic_verifier_response(prompt: str) -> Optional[str]:
    match = re.search(
        r"(?:^|\n)Query:\s*([\s\S]*?)\nAnswer:\s*([\s\S]*?)\nGT:\s*([^\n]*)\s*$",
        str(prompt or ""), re.IGNORECASE,
    )
    if not match:
        return None
    query, answer_text, gt_text = match.groups()
    pred_class, pred_coords = _parse_structured_answer(answer_text)
    gt_class, gt_coords = _parse_structured_answer(gt_text)
    if pred_class is None or gt_class is None:
        return "REJECT: answer is not a valid active canonical label or structured grounding output"
    if pred_class != gt_class:
        return "REJECT: canonical class mismatch"
    classification = _is_classification_context(query) or gt_coords is None
    if classification:
        return "ACCEPT"
    if pred_coords is None or gt_coords is None:
        return "REJECT: missing complete clockwise OBB"
    valid, reason = _valid_obb_values(pred_coords)
    if not valid:
        return f"REJECT: invalid predicted OBB ({reason})"
    pred_hbb, gt_hbb = _hbb_from_values(pred_coords), _hbb_from_values(gt_coords)
    iou = _hbb_iou_norm(pred_hbb, gt_hbb)
    px, py = (pred_hbb[0]+pred_hbb[2])/2, (pred_hbb[1]+pred_hbb[3])/2
    gx1, gy1, gx2, gy2 = gt_hbb
    mx = max(1.0, (gx2-gx1)*VERIFY_EXPANDED_GT_RATIO)
    my = max(1.0, (gy2-gy1)*VERIFY_EXPANDED_GT_RATIO)
    center_ok = gx1-mx <= px <= gx2+mx and gy1-my <= py <= gy2+my
    if iou < VERIFY_MIN_HBB_IOU or not center_ok:
        return f"REJECT: predicted region mismatches target geometry (HBB IoU={iou:.3f})"
    return "ACCEPT"


def _reasoner_format_status(
    text: str,
    must_final: bool,
    previous_questions: Optional[List[str]] = None,
    classification_task: bool = False,
    original_context: str = "",
    valid_perception_rounds: int = 0,
    valid_coordinate_rounds: int = 0,
) -> Tuple[bool, str]:
    if _looks_incomplete_response(text):
        return False, "incomplete_reasoner_response"
    thinking = _THINK_RE.search(text or "")
    questions = _QUESTION_RE.findall(text or "")
    final = _FINAL_RE.search(text or "")
    if thinking is None:
        return False, "missing_thinking"
    if _has_pathological_repetition(text):
        return False, "pathological_repetition"
    has_q = len(questions) == 1
    has_final = final is not None
    if has_q == has_final:
        return False, "must_contain_exactly_one_question_or_final"
    if len(questions) > 1:
        return False, "multiple_questions"
    if must_final and not has_final:
        return False, "last_round_requires_final"

    required_rounds = 1 if classification_task else 2
    if has_final and valid_perception_rounds < required_rounds:
        return False, (
            f"final_before_valid_perception_rounds:"
            f"{valid_perception_rounds}<{required_rounds}"
        )
    if has_final and not classification_task and valid_coordinate_rounds < 1:
        return False, "final_before_valid_coordinate_evidence"

    if has_q:
        repeated, score, _ = _duplicate_question(
            questions[0], list(previous_questions or [])
        )
        if repeated:
            return False, f"duplicate_or_similar_question:{score:.3f}"
        if classification_task and _classification_question_is_leading(questions[0]):
            return False, "classification_question_reveals_or_asks_class"
        unsafe_coordinate_question = _coordinate_question_is_unsafe(questions[0])
        if unsafe_coordinate_question:
            return False, unsafe_coordinate_question
        unrelated = _unrelated_question_roi(questions[0], original_context)
        if unrelated is not None:
            candidate, focus = unrelated
            return (
                False,
                "question_uses_unrelated_roi:"
                f"candidate={list(candidate)}:focus={list(focus)}",
            )
    return True, "ok"


def _coordinate_instruction(images: Optional[List[Tuple[Image.Image, str]]]) -> str:
    target = COORDINATE_TARGET.lower()
    zh = _QA_LANG == "zh"
    if target == "norm1000_obb":
        return (
            "使用 [0,1000] 归一化图像坐标，左上角为 (0,0)，右下角为 (1000,1000)。粗框返回 bbox_2d=[xmin,ymin,xmax,ymax]；旋转框返回顺时针 obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]。"
            if zh else
            "Use normalized image coordinates in [0,1000], with (0,0) at the top-left and (1000,1000) at the bottom-right. For a coarse box return bbox_2d=[xmin,ymin,xmax,ymax]. For a rotated box return clockwise obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]."
        )
    if target == "norm100_obb":
        return ("使用 [0,100] 归一化坐标，左上角为原点，只返回所需 bbox_2d 或顺时针 obb_8。" if zh else "Use normalized image coordinates in [0,100], with top-left origin. Return only the requested bbox_2d or clockwise obb_8.")
    if target == "pixel_obb":
        size = ""
        if images:
            width, height = images[0][0].size
            size = (f" 图像尺寸为 {width}×{height} 像素。" if zh else f" The image size is {width}x{height} pixels.")
        return (("使用原始图像像素坐标，左上角为原点。" + size + "只返回所需 bbox_2d 或顺时针 obb_8。") if zh else ("Use original-image pixel coordinates with top-left origin." + size + " Return only the requested bbox_2d or clockwise obb_8."))
    if target == "norm1000_hbb":
        return ("使用 [0,1000] 归一化坐标，左上角为原点，返回 bbox_2d=[xmin,ymin,xmax,ymax]。" if zh else "Use normalized image coordinates in [0,1000], with top-left origin, and return bbox_2d=[xmin,ymin,xmax,ymax].")
    return ("严格遵守原始任务中的坐标约定。" if zh else "Follow the coordinate convention from the original task exactly.")



def _redact_classification_candidates(text: str) -> str:
    """Hide the label list from the Perceiver while preserving target identity."""
    out = str(text or "")
    out = re.sub(
        r"请从以下标准类别中选择且只选择一个[:：][\s\S]*?(?=必须根据图像证据判断|只返回\s*classes\.txt|只返回classes\.txt)",
        "分类候选类别列表已对视觉感知模型隐藏。",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"(?:choose|select) exactly one (?:canonical )?label from[:：][\s\S]*?(?=use visual evidence|return only|respond only)",
        "The classification candidate list is hidden from the perception model. ",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _redact_roi_coordinates(text: str) -> str:
    """Hide coarse ROI numbers during crop-local coordinate estimation."""
    out = str(text or "")
    out = _FOCUS_ROI_CAPTURE_RE.sub("focus region [hidden]", out)
    out = _QUESTION_ROI_RE.sub("target ROI [hidden]", out)
    return out


def _focus_area_ratio(crop_meta: Optional[Dict[str, Any]]) -> float:
    if crop_meta is None:
        return 1.0
    x1, y1, x2, y2 = [float(v) for v in crop_meta["norm1000_bounds"]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1_000_000.0


def _select_perceiver_images(
    prepared_images: Optional[List[Tuple[Image.Image, str]]],
    crop_meta: Optional[Dict[str, Any]],
    original_context: str,
    atomic_question: str,
) -> Tuple[Optional[List[Tuple[Image.Image, str]]], str]:
    if not prepared_images or crop_meta is None or len(prepared_images) < 2:
        return prepared_images, "full_only"

    full_image = prepared_images[0]
    crop_image = prepared_images[-1]
    classification_task = _is_classification_context(original_context)
    coordinate_kind = "none" if classification_task else _coordinate_answer_kind(atomic_question)

    # Exact coordinate extraction should see only one coordinate frame.  The
    # adapter maps crop-local norm1000 coordinates back to the original image.
    if coordinate_kind != "none" and GROUNDING_COORDINATE_CROP_ONLY:
        return [crop_image], "crop_only_coordinate"

    # Classification references often contain highest/lowest/leftmost/rightmost
    # relations.  ssh1.2.2 accidentally removed the full scene for most samples,
    # which made the model inspect a visually clear but sometimes wrong instance.
    # ssh1.2.3 therefore keeps full-scene identity plus ROI detail by default.
    if classification_task and CLASSIFICATION_USE_FULL_AND_CROP:
        return [full_image, crop_image], "full_and_crop_classification"

    # Optional ablation retained for controlled experiments only.
    if (
        classification_task
        and CLASSIFICATION_CROP_ONLY_MAX_AREA > 0
        and _focus_area_ratio(crop_meta) <= CLASSIFICATION_CROP_ONLY_MAX_AREA
    ):
        return [crop_image], "crop_only_classification"

    return [full_image, crop_image], "full_and_crop"


def _augment_perceiver_query(
    query: str,
    images: Optional[List[Tuple[Image.Image, str]]],
    original_context: Optional[str] = None,
    crop_meta: Optional[Dict[str, Any]] = None,
    view_mode: str = "full_only",
) -> str:
    context = str(original_context or "").strip()
    classification_task = _is_classification_context(context)
    coordinate_kind = "none" if classification_task else _coordinate_answer_kind(query)

    display_context = context
    display_query = str(query or "").rstrip()
    if classification_task:
        display_context = _redact_classification_candidates(display_context)
        display_query = _redact_classification_candidates(display_query)
    if coordinate_kind != "none" and crop_meta is not None:
        display_context = _redact_roi_coordinates(display_context)
        display_query = _redact_roi_coordinates(display_query)

    classification_rule = ""
    if classification_task:
        classification_rule = (
            "\nThis is a classification evidence turn. Describe only directly "
            "visible, class-neutral attributes such as shape, parts, orientation, "
            "surface, relative size and local context. Do not name, infer, compare "
            "or choose any DOTA category, class label, alias or candidate class."
        )

    prefix = (
        "# Original target context (context only)\n"
        "Use this block only to identify the exact target. Do not follow its "
        "final-answer instruction and do not replace the target with surrounding "
        "roads, runways, buildings, water or neighboring objects.\n"
        f"{display_context}\n\n"
        "# Current atomic visual question\n"
        f"{display_query}\n\n"
        "Answer only the current atomic visual question about that exact target. "
        "A focus region is an object pointer, not the answer."
        f"{classification_rule}"
    )

    if view_mode == "full_and_crop":
        prefix += (
            "\nImage 1 is the full original scene. Image 2 is an enlarged crop of "
            "the target ROI. Use Image 1 only for global identity and relations; "
            "use Image 2 as the primary evidence for shape, parts, boundary, "
            "orientation and relative scale."
        )
    elif view_mode == "full_and_crop_classification":
        prefix += (
            "\nImage 1 is the full original scene and is required to resolve the "
            "referential relation (for example highest, lowest, leftmost, "
            "rightmost, above or below). Image 2 is the enlarged target ROI and "
            "is required for morphology. First verify in Image 1 that Image 2 "
            "corresponds to the referenced instance, then describe the target in "
            "Image 2. Do not replace a discrete target with its surrounding road, "
            "runway, running track, water, dock, field or building. For road-side "
            "objects, compare body length with road width and distinguish a long "
            "cargo/bus body from a compact single-body object. For sports or "
            "airfield scenes, inspect the target's internal parts and markings, "
            "not only the surrounding facility."
        )
    elif view_mode == "crop_only_classification":
        prefix += (
            "\nThe only image is an enlarged crop centered on the pointed target. "
            "Judge the target itself, not the crop border or surrounding road. For "
            "vehicle evidence, compare body length with road width and look for a "
            "long cargo/bus body versus a compact single-body car."
        )
    elif view_mode == "crop_only_coordinate":
        prefix += (
            "\nThe only image is the enlarged target ROI. It defines the sole "
            "coordinate frame for this turn. Inspect the visible object boundary "
            "inside the crop; do not use or reconstruct hidden coarse ROI numbers."
        )

    if coordinate_kind == "none":
        return prefix

    crop_coordinate_rule = ""
    if crop_meta is not None and MAP_CROP_COORDINATES:
        crop_coordinate_rule = (
            "Measure only the exact object in the provided ROI image. Treat its "
            "top-left as (0,0) and bottom-right as (1000,1000). Return crop-local "
            "normalized coordinates; the adapter will map them back to the full "
            "image. Do not return the crop boundary. "
        )
    return (
        f"{prefix}\n\n"
        f"{crop_coordinate_rule}"
        "Coordinate reminder: estimate coordinates from visible pixels; no physical "
        "map scale is required. "
        f"{_coordinate_instruction(images)} "
        "Estimate only the single referred target. "
        f"{_coordinate_contract(coordinate_kind)}"
    )


def _final_turn_instruction(round_index: int) -> str:
    if _QA_LANG == "zh":
        return f"""

# 强制最终轮
这是 Reasoner 第 {round_index}/{MAX_LOOP} 轮，也是最后允许轮次。
不要再输出 <question>。使用已有证据，遵守原问题坐标制和精确答案格式，只输出：
<thinking>简短证据总结与格式检查</thinking>
[Final Answer]: <原问题要求的直接答案>
最终答案后不得添加文字。
""".rstrip()
    return f"""

# Mandatory final-turn override
This is Reasoner round {round_index}/{MAX_LOOP}, the final allowed round.
Do not output another <question>. Use the evidence already collected and obey the original query's coordinate convention and exact answer format. Output exactly:
<thinking>brief evidence summary and final format check</thinking>
[Final Answer]: <direct answer required by the original query>
Do not add any text after the final answer.
""".rstrip()



def _repair_instruction(
    query: str,
    invalid_response: str,
    reason: str,
    must_final: bool,
    previous_questions: Optional[List[str]] = None,
    classification_task: bool = False,
    original_context: str = "",
) -> str:
    target = (
        "a final answer, not another question" if must_final
        else "exactly one atomic question or a final answer"
    )
    history = list(previous_questions or [])
    history_block = (
        "\n".join(f"{index + 1}. {question}" for index, question in enumerate(history))
        if history
        else "(none)"
    )
    task_rules = ""
    if classification_task:
        task_rules = """
If asking a question, it must be class-neutral:
- do not use any active canonical category name or class label;
- do not ask what class/category/label the target is;
- do not offer candidates, comparisons, or yes/no guesses such as
  "is it a bridge?" or "vehicle versus storage tank?";
- ask for one directly visible attribute only: shape, parts, orientation,
  surface, support, scale, or immediate local context.
These restrictions apply to <question> content, not to a final answer.
""".strip()
    else:
        focus = _extract_focus_roi(original_context)
        focus_rule = (
            f"The immutable original focus ROI is {list(focus)}. "
            if focus is not None
            else ""
        )
        task_rules = (
            f"{focus_rule}Keep the question bound to that exact target. "
            "Do not introduce a different numeric search region. If asking for "
            "coordinates, request one complete bbox_2d or one complete clockwise "
            "obb_8, not separate corners."
        )
    reason_rules = ""
    if reason.startswith("duplicate_or_similar_question"):
        reason_rules = (
            "The new question must investigate a genuinely different atomic fact "
            "from every history item below. Merely changing wording, a direction "
            "word, or numeric bounds is still repetition. If no new visual fact is "
            "needed, finalize instead."
        )
    elif reason.startswith("question_uses_unrelated_roi"):
        reason_rules = (
            "Do not use the rejected ROI. Use the immutable original focus ROI "
            "verbatim, ask about the already-pointed target without a new numeric "
            "ROI, or finalize."
        )
    elif reason == "classification_question_reveals_or_asks_class":
        reason_rules = (
            "Remove every class/category name, alias, candidate comparison and "
            "yes/no class guess from the question."
        )
    elif reason.startswith("final_before_valid_perception_rounds"):
        reason_rules = (
            "Do not finalize yet. Ask one new atomic visual question that obtains "
            "missing evidence from the exact target."
        )
    elif reason == "final_before_valid_coordinate_evidence":
        reason_rules = (
            "Do not finalize. Ask for one complete tight clockwise OBB of the exact "
            "target in a single coordinate question; do not ask separate corners."
        )
    elif reason == "partial_corner_coordinate_question":
        reason_rules = (
            "Never ask for one corner. Ask once for all four distinct non-collinear "
            "clockwise OBB corners and require only an obb_8 response."
        )
    elif reason == "incomplete_reasoner_response":
        reason_rules = (
            "Regenerate a complete response with fully closed structural tags. "
            "Do not end at a partial word or partial XML tag."
        )
    compact_query = _compact_reasoner_query(query, original_context, 5200)
    invalid_excerpt = "(omitted because the response was truncated)" if reason == "finish_reason=length" else invalid_response[:700]
    return f"""
{compact_query}

# Strict response repair
Invalid because: {reason}.
Invalid-response excerpt: {invalid_excerpt}
Previous questions: {history_block}
Task rules: {task_rules}
{reason_rules}

Regenerate only one concise response for the same task. Begin with one
<thinking>...</thinking> block of at most 90 words, then output {target}.
A question must be one <question>...</question> of at most 35 words.
A final must be one [Final Answer]: line. Never repeat a sentence or prior question.
""".strip()


def _perceiver_repair_instruction(
    effective_query: str,
    invalid_response: str,
    reason: str,
    atomic_question: str,
    original_context: str,
) -> str:
    classification_task = _is_classification_context(original_context)
    coordinate_kind = (
        "none" if classification_task else _coordinate_answer_kind(atomic_question)
    )
    rules: List[str] = []
    if classification_task:
        rules.append(
            "Return only directly visible, class-neutral attributes. Do not name, "
            "infer, compare, or choose any DOTA class, category, label, alias, "
            "subtype, or candidate."
        )
    if coordinate_kind != "none":
        rules.append(_coordinate_contract(coordinate_kind))
        rules.append(
            "Inspect the object boundary again. Use four distinct non-collinear "
            "corners. The box must be substantially tighter than the entire ROI "
            "unless the visible target truly fills nearly all of it."
        )
    if reason == "coordinate_copies_entire_crop":
        rules.append(
            "The previous answer copied the ROI boundary. Ignore all crop edges and "
            "place corners on the actual target silhouette inside the crop."
        )
    elif reason in {"obb_duplicate_points", "obb_zero_or_tiny_area", "obb_self_intersection"}:
        rules.append(
            "The previous four points did not form a valid quadrilateral. Re-estimate "
            "all four physical corners independently around the target."
        )
    elif reason == "unsolicited_coordinate_payload":
        rules.append(
            "The current question did not request coordinates. Remove every bbox, "
            "OBB and class|coordinate payload; answer only the requested visual fact."
        )
    if not rules:
        rules.append(
            "Answer only the current atomic visual question about the exact target."
        )
    invalid_excerpt = (
        "(omitted because the response was truncated)"
        if reason == "finish_reason=length"
        else str(invalid_response or "")[:500]
    )
    return f"""
{effective_query[-5200:].rstrip()}

# Perceiver response repair
Rejected because: {reason}.
Invalid-response excerpt: {invalid_excerpt}

{' '.join(rules)}
Do not repeat the original task, do not switch targets, and do not add a
self-dialogue or final task answer.
""".strip()


def _safe_perceiver_failure_response(
    reason: str,
    coordinate_kind: str,
    classification_task: bool,
) -> str:
    if _QA_LANG == "zh":
        if coordinate_kind != "none":
            return (
                "前一坐标响应无效，不能作为证据。下一轮必须重新请求放大ROI中"
                "精确目标的完整紧致顺时针OBB，并只接受obb_8结构。"
            )
        if classification_task:
            return (
                "前一视觉观察不够可靠或不够类别中性。下一轮请询问精确目标的"
                "另一个直接可见属性，且不得说出候选类别。"
            )
        return "前一视觉观察不可靠。请针对同一精确目标询问一个更具体、尚未问过的视觉事实。"
    if coordinate_kind != "none":
        return (
            "The coordinate estimate was invalid and must not be used. Ask again "
            "for one complete tight clockwise OBB of the exact target in the "
            "enlarged ROI."
        )
    if classification_task:
        return (
            "The previous observation was not sufficiently class-neutral or reliable. "
            "Ask for one different directly visible attribute of the exact target."
        )
    return (
        "The previous visual observation was unreliable. Ask a more specific visual "
        "question about the exact target before finalizing."
    )


def _question_dimension(question: str) -> str:
    low = re.sub(r"\s+", " ", str(question or "").casefold())
    dimensions = (
        ("shape", ("shape", "outline", "geometry", "boundary", "form", "形状", "轮廓", "边界", "几何")),
        ("parts", ("part", "wing", "tail", "deck", "roof", "support", "部件", "结构", "机翼", "尾部")),
        ("orientation", ("orient", "direction", "axis", "horizontal", "vertical", "diagonal", "方向", "朝向", "长轴", "水平", "垂直", "倾斜")),
        ("scale", ("size", "scale", "length", "width", "relative", "larger", "smaller", "大小", "尺度", "长度", "宽度")),
        ("surface", ("surface", "texture", "color", "material", "reflective", "表面", "纹理", "颜色", "材质")),
        ("markings", ("marking", "line", "stripe", "lane", "goal", "net", "标记", "线条", "条纹", "分区")),
        ("context", ("surround", "adjacent", "beneath", "support", "road", "water", "paved", "周围", "相邻", "下方", "道路", "水面", "铺装")),
    )
    for name, terms in dimensions:
        if any(term in low for term in terms):
            return name
    return "other"


def _choose_unasked_question(
    candidates: List[str], previous_questions: List[str]
) -> Optional[str]:
    for candidate in candidates:
        repeated, _, _ = _duplicate_question(candidate, previous_questions)
        if not repeated:
            return candidate
    return None


def _safe_reasoner_fallback(
    previous_questions: List[str],
    classification_task: bool,
    reason: str,
) -> Optional[str]:
    """Use a bounded bank of non-repeating recovery questions.

    Returning ``None`` after exhaustion prevents the adapter from injecting the
    same generic question indefinitely. The official loop will then preserve the
    last rejected response and the run will fail closed rather than fabricate
    additional evidence.
    """
    coordinate_recovery = reason == "final_before_valid_coordinate_evidence" or any(
        marker in str(reason or "")
        for marker in (
            "coordinate", "obb", "bbox", "corner", "角点", "坐标",
            "partial_corner", "invalid_perceiver",
        )
    )
    if _QA_LANG == "zh":
        if classification_task:
            candidates = [
                "精确目标直接可见的整体轮廓和二维几何形状是什么？",
                "精确目标的前部、中部和后部在轮廓上如何分段？",
                "精确目标顶部是否存在与主体明显不同的设备或承载结构？",
                "精确目标相对于紧邻承载表面的大小和长宽比例如何？",
                "精确目标内部可见哪些标记、分区或重复图案？",
                "精确目标的表面纹理及其与背景的边界对比如何？",
                "精确目标在图像中的主轴方向是什么？",
            ]
        elif coordinate_recovery:
            candidates = [
                "放大ROI中精确目标的四个不同顺时针OBB角点是什么？只返回obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]。",
                "忽略粗略搜索框和裁剪边缘，沿精确目标真实外轮廓重新估计完整紧致OBB；只返回obb_8=[x1,y1,x2,y2,x3,y3,x4,y4]。",
                "请重新独立估计精确目标的四个物理角点；四点不得重复、共线、自交或覆盖整个ROI，只返回obb_8结构。",
                "以放大ROI左上角为(0,0)、右下角为(1000,1000)，给出精确目标完整顺时针OBB；不要逐角回答。",
                "仅检查精确目标本体，不要使用道路、阴影或ROI边缘；返回紧贴目标的四点顺时针obb_8。",
                "重新观察目标四条实际边界的交点并一次性返回完整obb_8，禁止返回单个角点或自然语言解释。",
            ]
        else:
            candidates = [
                "精确目标最显著且独立于背景的二维轮廓特征是什么？",
                "精确目标可见哪些与周围对象不同的结构部件？",
                "精确目标的主轴方向和完整车体宽度如何？",
                "精确目标与紧邻道路、地面或其他对象之间的边界在哪里？",
            ]
        question = _choose_unasked_question(candidates, previous_questions)
        if question is None:
            return None
        return (
            "<thinking>前一响应在有限修复后仍不合法。保持同一目标，并从有限恢复问题中选择一个尚未问过的问题。</thinking>\n"
            f"<question>{question}</question>"
        )

    if classification_task:
        candidates = [
            "What is the exact target's visible outline and two-dimensional geometry?",
            "How do the front, middle, and rear portions of the exact target differ in outline?",
            "Which distinct structural part is directly visible on top of the exact target?",
            "What are the target's relative size and length-to-width proportion?",
            "Which internal markings, divisions, or repeated patterns are visible?",
            "What surface texture and target-background boundary contrast are visible?",
            "What is the exact target's principal axis orientation?",
        ]
    elif coordinate_recovery:
        candidates = [
            "What are the four distinct clockwise OBB corners of the exact target in the enlarged ROI? Return only obb_8=[x1,y1,x2,y2,x3,y3,x4,y4].",
            "Ignore the coarse search box and crop edges; re-estimate one complete tight clockwise OBB on the actual target silhouette and return only obb_8.",
            "Independently re-estimate all four physical target corners; do not repeat points, return a line, cross edges, or cover the full ROI. Return only obb_8.",
            "Using the enlarged ROI coordinate frame, return the complete tight clockwise target OBB in one response; never answer one corner at a time.",
            "Inspect only the exact target body, not roads, shadows, or ROI edges, and return four tight clockwise OBB corners.",
            "Re-observe the intersections of the four actual target boundaries and return one complete obb_8 without prose.",
        ]
    else:
        candidates = [
            "What two-dimensional outline most clearly separates the exact target from the background?",
            "Which visible structural part distinguishes the exact target from nearby objects?",
            "What are the target's principal orientation and complete body width?",
            "Where is the visible boundary between the exact target and the adjacent surface?",
        ]
    question = _choose_unasked_question(candidates, previous_questions)
    if question is None:
        return None
    return (
        "<thinking>The prior response remained invalid after bounded repair. I will keep the same target and use one unasked question from a finite recovery bank.</thinking>\n"
        f"<question>{question}</question>"
    )



class APIModel:
    """Drop-in replacement for the official APIModel class."""

    def __init__(self, model_name: str, system_prompt: Optional[str] = None):
        if model_name not in MODEL_ROUTE:
            raise NotImplementedError(
                f"No local route configured for official model name: {model_name}"
            )
        self.model_name = model_name
        self.raw_system_prompt = system_prompt or ""
        low_prompt = self.raw_system_prompt.lower()
        if "instruction rewriter" in low_prompt:
            self.role = "rewriter"
        elif model_name == "gpt-5-mini":
            self.role = "reasoner"
        elif model_name == "gemini-2.5-flash":
            self.role = "perceiver"
        else:
            self.role = "verifier"

        profile = (
            os.getenv("ALS_PROMPT_PROFILE", "official_general") or "official_general"
        ).strip()
        if profile.lower() in {"true", "false", "1", "0", "yes", "no"}:
            raise RuntimeError(
                "ALS_PROMPT_PROFILE resolved to a boolean-like value "
                f"{profile!r}. Use the v4.2.1+ NUL-delimited launcher and set "
                "trajectory.prompt_profile to a profile such as 'obb_grounding_v2'."
            )
        self.profile = profile
        self.system_prompt = (
            apply_overlay(system_prompt, model_name, profile)
            if apply_overlay
            else system_prompt
        )
        self.base_url, self.served_name = MODEL_ROUTE[model_name]

    def _single_call(
        self,
        query: str,
        images: Optional[List[Tuple[Image.Image, str]]] = None,
        *,
        call_kind: str = "normal",
        repair_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
            timeout=REQUEST_TIMEOUT,
        )
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if images:
            content: List[Dict[str, Any]] = []
            for image, suffix in images:
                encoded = _encode_image(image, suffix)
                mime = (
                    "jpeg"
                    if (suffix or "jpeg").lower() in {"jpg", "jpeg"}
                    else (suffix or "png").lower()
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{encoded}"},
                    }
                )
            content.append({"type": "text", "text": query})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": query})

        max_tokens, temperature, top_p = _sampling(self.model_name)
        request_kwargs: Dict[str, Any] = {
            "model": self.served_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if OMNI_TEXT_ONLY:
            # ``extra_body`` remains compatible with OpenAI-compatible servers
            # while forwarding the vLLM-Omni-specific modality control.
            request_kwargs["extra_body"] = {"modalities": ["text"]}
        result = client.chat.completions.create(**request_kwargs)
        choice = result.choices[0]
        content_text = choice.message.content or ""
        usage = getattr(result, "usage", None)
        meta: Dict[str, Any] = {
            "event": "api_call",
            "role": self.role,
            "official_name": self.model_name,
            "served_name": self.served_name,
            "call_kind": call_kind,
            "repair_index": repair_index,
            "finish_reason": getattr(choice, "finish_reason", None),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "query_chars": len(query),
            "response_chars": len(content_text),
            "response_preview": content_text[:300],
            "omni_text_only": OMNI_TEXT_ONLY,
        }
        if self.role == "perceiver":
            meta.update(
                {
                    "original_context_attached": "# Original target context" in query,
                    "focus_roi_attached": _extract_focus_roi(_current_reasoner_context()) is not None,
                    "image_count": len(images or []),
                    "perceiver_view_mode": str(
                        getattr(_THREAD_CONTEXT, "perceiver_view_mode", "full_only") or "full_only"
                    ),
                    "focus_crop_attached": str(
                        getattr(_THREAD_CONTEXT, "perceiver_view_mode", "full_only") or "full_only"
                    ) != "full_only",
                    "original_context_sha256": str(
                        getattr(_THREAD_CONTEXT, "original_target_sha256", "") or ""
                    ),
                    "original_context_task": str(
                        getattr(_THREAD_CONTEXT, "original_target_task", "") or ""
                    ),
                }
            )
        if usage is not None:
            meta.update(
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            )
        _write_log(meta)
        return content_text, meta

    def _call_with_retries(
        self,
        query: str,
        images: Optional[List[Tuple[Image.Image, str]]],
        *,
        call_kind: str,
        repair_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._single_call(
                    query,
                    images,
                    call_kind=call_kind,
                    repair_index=repair_index,
                )
            except Exception as exc:
                last_error = exc
                _write_log(
                    {
                        "event": "api_error",
                        "role": self.role,
                        "official_name": self.model_name,
                        "served_name": self.served_name,
                        "call_kind": call_kind,
                        "attempt": attempt,
                        "max_retries": MAX_RETRIES,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"[LOCAL API RETRY] role={self.role} official_name={self.model_name} "
                    f"served_name={self.served_name} attempt={attempt}/{MAX_RETRIES} "
                    f"failed: {exc}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * attempt)
        assert last_error is not None
        raise last_error

    def get_response(
        self,
        query: str,
        images: Optional[List[Tuple[Image.Image, str]]] = None,
    ) -> str:
        if (
            self.role == "rewriter"
            and BYPASS_STRUCTURED_REWRITE
            and _is_structured_task(query)
        ):
            original = _extract_original_query(query)
            _write_log(
                {
                    "event": "structured_rewrite_bypass",
                    "role": self.role,
                    "official_name": self.model_name,
                    "served_name": self.served_name,
                    "query_chars": len(query),
                    "response_chars": len(original),
                }
            )
            return original

        if self.role == "verifier" and DETERMINISTIC_STRUCTURED_VERIFIER:
            verdict = _deterministic_verifier_response(query)
            if verdict is not None:
                _write_log({
                    "event": "deterministic_verifier",
                    "role": self.role,
                    "accepted": verdict == "ACCEPT",
                    "response": verdict,
                })
                return verdict

        effective_query = query
        effective_images = images
        crop_meta: Optional[Dict[str, Any]] = None
        perceiver_view_mode = "full_only"
        must_final = False
        round_index: Optional[int] = None
        previous_questions: List[str] = []
        classification_task = False
        original_context = ""

        if self.role == "perceiver":
            original_context = _current_reasoner_context()
            if REQUIRE_PERCEIVER_CONTEXT and not original_context:
                _write_log(
                    {
                        "event": "perceiver_context_missing",
                        "role": self.role,
                        "query_chars": len(query),
                    }
                )
                raise RuntimeError(
                    "Perceiver call has no original target context; refusing unsafe atomic-only vision call"
                )
            prepared_images, crop_meta = _build_focus_crop(images, original_context)
            effective_images, perceiver_view_mode = _select_perceiver_images(
                prepared_images,
                crop_meta,
                original_context,
                query,
            )
            _THREAD_CONTEXT.perceiver_view_mode = perceiver_view_mode
            effective_query = _augment_perceiver_query(
                query,
                effective_images,
                original_context,
                crop_meta,
                perceiver_view_mode,
            )
        elif self.role == "reasoner":
            original_context = _remember_reasoner_context(query)
            classification_task = _is_classification_context(original_context)
            previous_questions = [
                re.sub(r"\s+", " ", value).strip()
                for value in _QUESTION_RE.findall(query)
            ]
            round_index = _infer_reasoner_round(query)
            valid_perception_rounds, valid_coordinate_rounds, _ = _current_evidence_counts()
            required_rounds = 1 if classification_task else 2
            evidence_ready = (
                valid_perception_rounds >= required_rounds
                and (classification_task or valid_coordinate_rounds >= 1)
            )
            must_final = (
                FORCE_FINAL_ON_LAST_ROUND
                and round_index >= MAX_LOOP
                and evidence_ready
            )
            if must_final:
                effective_query = query.rstrip() + _final_turn_instruction(round_index)
            effective_query = _compact_reasoner_query(
                effective_query, original_context, REASONER_COMPACT_MAX_CHARS
            )

        response, meta = self._call_with_retries(
            effective_query,
            effective_images,
            call_kind="normal",
        )

        if self.role == "perceiver":
            classification_task = _is_classification_context(original_context)
            coordinate_kind = (
                "none" if classification_task else _coordinate_answer_kind(query)
            )
            response, normalized, normalization_reason = _canonicalize_perceiver_coordinate_response(
                response, coordinate_kind, crop_meta
            )
            if normalized:
                _write_log({
                    "event": "perceiver_coordinate_canonicalized",
                    "role": self.role,
                    "coordinate_kind": coordinate_kind,
                    "crop_mapped": bool(MAP_CROP_COORDINATES and crop_meta),
                    "view_mode": perceiver_view_mode,
                })
            valid, reason = _perceiver_response_status(response, query, original_context)
            if not valid and normalization_reason != "ok":
                reason = normalization_reason
            truncated = str(meta.get("finish_reason") or "").lower() == "length"
            if valid and not truncated:
                _record_perceiver_evidence(coordinate_kind)
                return response

            last_response = response
            last_reason = "finish_reason=length" if truncated else reason
            for repair_index in range(1, MAX_REPAIR_ATTEMPTS + 1):
                repair_query = _perceiver_repair_instruction(
                    effective_query=effective_query,
                    invalid_response=last_response,
                    reason=last_reason,
                    atomic_question=query,
                    original_context=original_context,
                )
                repaired, repair_meta = self._call_with_retries(
                    repair_query,
                    effective_images,
                    call_kind="perceiver_response_repair",
                    repair_index=repair_index,
                )
                repaired, repaired_normalized, repaired_normalization_reason = (
                    _canonicalize_perceiver_coordinate_response(
                        repaired, coordinate_kind, crop_meta
                    )
                )
                if repaired_normalized:
                    _write_log({
                        "event": "perceiver_coordinate_canonicalized",
                        "role": self.role,
                        "coordinate_kind": coordinate_kind,
                        "crop_mapped": bool(MAP_CROP_COORDINATES and crop_meta),
                        "view_mode": perceiver_view_mode,
                        "repair_index": repair_index,
                    })
                repaired_valid, repaired_reason = _perceiver_response_status(
                    repaired, query, original_context
                )
                if not repaired_valid and repaired_normalization_reason != "ok":
                    repaired_reason = repaired_normalization_reason
                repaired_truncated = (
                    str(repair_meta.get("finish_reason") or "").lower() == "length"
                )
                if repaired_valid and not repaired_truncated:
                    _write_log({
                        "event": "perceiver_response_repaired",
                        "role": self.role,
                        "reason": last_reason,
                        "repair_attempts": repair_index,
                        "view_mode": perceiver_view_mode,
                    })
                    _record_perceiver_evidence(coordinate_kind)
                    return repaired
                last_response = repaired
                last_reason = (
                    "finish_reason=length" if repaired_truncated else repaired_reason
                )

            safe_response = _safe_perceiver_failure_response(
                last_reason,
                coordinate_kind,
                classification_task,
            )
            _write_log({
                "event": "perceiver_response_unrepaired",
                "role": self.role,
                "reason": last_reason,
                "repair_attempts": MAX_REPAIR_ATTEMPTS,
                "view_mode": perceiver_view_mode,
                "fail_closed": PERCEIVER_FAIL_CLOSED,
            })
            return safe_response if PERCEIVER_FAIL_CLOSED else last_response

        if self.role != "reasoner":
            return response

        response, canonicalized = _canonicalize_reasoner_response(response)
        if canonicalized:
            _write_log(
                {
                    "event": "reasoner_format_canonicalized",
                    "role": self.role,
                    "round": round_index,
                }
            )

        valid, reason = _reasoner_format_status(
            response,
            must_final,
            previous_questions,
            classification_task,
            original_context,
            valid_perception_rounds,
            valid_coordinate_rounds,
        )
        truncated = str(meta.get("finish_reason") or "").lower() == "length"
        if valid and not truncated:
            return response

        repair_reason = "finish_reason=length" if truncated else reason
        for repair_index in range(1, MAX_REPAIR_ATTEMPTS + 1):
            repair_query = _repair_instruction(
                query=effective_query,
                invalid_response=response,
                reason=repair_reason,
                must_final=must_final,
                previous_questions=previous_questions,
                classification_task=classification_task,
                original_context=original_context,
            )
            response, meta = self._call_with_retries(
                repair_query,
                effective_images,
                call_kind="format_repair",
                repair_index=repair_index,
            )
            response, canonicalized = _canonicalize_reasoner_response(response)
            if canonicalized:
                _write_log(
                    {
                        "event": "reasoner_format_canonicalized",
                        "role": self.role,
                        "round": round_index,
                        "repair_index": repair_index,
                    }
                )
            valid, reason = _reasoner_format_status(
                response,
                must_final,
                previous_questions,
                classification_task,
                original_context,
                valid_perception_rounds,
                valid_coordinate_rounds,
            )
            truncated = str(meta.get("finish_reason") or "").lower() == "length"
            if valid and not truncated:
                _write_log(
                    {
                        "event": "reasoner_format_repaired",
                        "role": self.role,
                        "round": round_index,
                        "repair_index": repair_index,
                    }
                )
                return response
            repair_reason = "finish_reason=length" if truncated else reason

        fallback = None
        if REASONER_FAIL_CLOSED and not must_final:
            fallback = _safe_reasoner_fallback(
                previous_questions, classification_task, repair_reason
            )
            if fallback is None:
                _write_log({
                    "event": "reasoner_fallback_bank_exhausted",
                    "role": self.role,
                    "round": round_index,
                    "reason": repair_reason,
                })
        _write_log(
            {
                "event": "reasoner_format_unrepaired",
                "role": self.role,
                "round": round_index,
                "reason": repair_reason,
                "repair_attempts": MAX_REPAIR_ATTEMPTS,
                "fail_closed": bool(fallback),
            }
        )
        return fallback if fallback is not None else response
