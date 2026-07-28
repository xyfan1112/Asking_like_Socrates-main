#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # OpenCV is preferred but the core geometry has a pure NumPy fallback.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal CPU environments
    cv2 = None

DOTA_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ALIASES = {
    "storage-tank": "storage tank", "baseball-diamond": "baseball diamond",
    "tennis-court": "tennis court", "basketball-court": "basketball court",
    "ground-track-field": "ground track field", "large-vehicle": "large vehicle",
    "small-vehicle": "small vehicle", "soccer-ball-field": "soccer ball field",
    "swimming-pool": "swimming pool",
    # Natural visual subtypes used by generic VLMs.
    "bus": "large vehicle", "truck": "large vehicle", "lorry": "large vehicle",
    "trailer": "large vehicle", "truck trailer": "large vehicle",
    "tractor trailer": "large vehicle", "heavy vehicle": "large vehicle",
    "car": "small vehicle", "sedan": "small vehicle", "suv": "small vehicle",
    "pickup": "small vehicle", "pickup truck": "small vehicle", "compact van": "small vehicle",
    "airplane": "plane", "aircraft": "plane", "jet": "plane",
    "boat": "ship", "vessel": "ship",
}


def load_settings(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("project", "paths", "models", "agents", "trajectory", "training", "evaluation"):
        if key not in data:
            raise ValueError(f"settings 缺少顶层字段: {key}")
    return data


def root_from_script(script_file: str | Path) -> Path:
    p = Path(script_file).resolve()
    for parent in p.parents:
        if (parent / 'settings.json').is_file() or (parent / 'settings.example.json').is_file():
            return parent
    return p.parents[1]


def settings_from_cli(script_file: str | Path, value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else root_from_script(script_file) / 'settings.json'


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: str | Path, skip_bad: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not skip_bad:
                    raise ValueError(f"{path} 第 {line_no} 行 JSON 损坏")
    return rows


def stable_id(*parts: Any) -> str:
    return hashlib.sha1("||".join(str(x) for x in parts).encode("utf-8")).hexdigest()


def normalize_class_name(text: str | None) -> str | None:
    name = str(text or "").strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip(" .,:;[](){}\"'")
    name = ALIASES.get(name, name)
    name = name.replace("-", " ")
    name = ALIASES.get(name, name)
    return name if name in DOTA_CLASSES else None


def polygon_area(points: Iterable[Iterable[float]]) -> float:
    arr = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if len(arr) < 3:
        return 0.0
    if cv2 is not None:
        return abs(float(cv2.contourArea(arr)))
    x, y = arr[:, 0], arr[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def order_polygon(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if len(arr) != 4:
        raise ValueError("OBB 必须包含四个点")
    c = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    return arr[np.argsort(angles)]


def obb_iou(a: Iterable[Iterable[float]], b: Iterable[Iterable[float]]) -> float:
    pa, pb = order_polygon(a), order_polygon(b)
    area_a, area_b = polygon_area(pa), polygon_area(pb)
    if area_a <= 1e-8 or area_b <= 1e-8:
        return 0.0
    if cv2 is not None:
        inter, _ = cv2.intersectConvexConvex(cv2.convexHull(pa), cv2.convexHull(pb))
        intersection = max(0.0, float(inter))
    else:
        intersection = polygon_area(_convex_intersection(pa, pb))
    union = area_a + area_b - intersection
    return intersection / union if union > 1e-8 else 0.0


def _convex_intersection(
    subject: Iterable[Iterable[float]],
    clip: Iterable[Iterable[float]],
) -> np.ndarray:
    """Sutherland-Hodgman convex polygon clipping fallback."""
    output = np.asarray(list(subject), dtype=np.float64).reshape(-1, 2)
    clip_arr = np.asarray(list(clip), dtype=np.float64).reshape(-1, 2)
    if len(output) < 3 or len(clip_arr) < 3:
        return np.empty((0, 2), dtype=np.float64)
    signed = float(
        np.dot(clip_arr[:, 0], np.roll(clip_arr[:, 1], -1))
        - np.dot(clip_arr[:, 1], np.roll(clip_arr[:, 0], -1))
    )
    orientation = 1.0 if signed >= 0 else -1.0

    def cross(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
        return float(np.cross(b - a, p - a))

    def inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return orientation * cross(a, b, p) >= -1e-9

    def intersection(
        start: np.ndarray,
        end: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        direction_1 = end - start
        direction_2 = b - a
        denominator = float(np.cross(direction_1, direction_2))
        if abs(denominator) <= 1e-12:
            return end
        t = float(np.cross(a - start, direction_2)) / denominator
        return start + t * direction_1

    for index, a in enumerate(clip_arr):
        b = clip_arr[(index + 1) % len(clip_arr)]
        input_vertices = output
        if len(input_vertices) == 0:
            break
        clipped: list[np.ndarray] = []
        start = input_vertices[-1]
        for end in input_vertices:
            if inside(end, a, b):
                if not inside(start, a, b):
                    clipped.append(intersection(start, end, a, b))
                clipped.append(end)
            elif inside(start, a, b):
                clipped.append(intersection(start, end, a, b))
            start = end
        output = (
            np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
            if clipped
            else np.empty((0, 2), dtype=np.float64)
        )
    return output


def find_label(image_path: Path, split_root: Path) -> Path | None:
    for p in (split_root / "labels" / f"{image_path.stem}.txt", image_path.with_suffix(".txt")):
        if p.is_file():
            return p.resolve()
    return None


def discover_images(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / split / "images"
    if not image_dir.is_dir():
        return []
    out = []
    for image in sorted(image_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            label = find_label(image, dataset_root / split)
            if label is None:
                raise FileNotFoundError(f"缺少标签: {image}")
            out.append((image.resolve(), label))
    return out


def load_obb_label(label_path: Path, width: int, height: int) -> tuple[list[dict[str, Any]], str]:
    objects, modes = [], set()
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 9:
            raise ValueError(f"{label_path}:{line_no} 需要 9 列，实际 {len(parts)}")
        class_id = int(float(parts[0]))
        if not 0 <= class_id < len(DOTA_CLASSES):
            raise ValueError(f"{label_path}:{line_no} class_id={class_id} 越界")
        vals = [float(x) for x in parts[1:]]
        normalized = max(abs(x) for x in vals) <= 1.5
        modes.add("normalized" if normalized else "pixel")
        pts = []
        for i in range(0, 8, 2):
            x, y = vals[i], vals[i + 1]
            if normalized:
                x, y = x * width, y * height
            pts.append([float(x), float(y)])
        cx = sum(x for x, _ in pts) / 4
        cy = sum(y for _, y in pts) / 4
        objects.append({
            "object_index": len(objects), "class_id": class_id,
            "class_name": DOTA_CLASSES[class_id], "points_pixel": pts,
            "center_pixel": [cx, cy], "area_pixel": polygon_area(pts),
        })
    return objects, next(iter(modes)) if len(modes) == 1 else "mixed"



def inspect_obb_object(
    obj: dict[str, Any],
    width: int,
    height: int,
    min_area: float = 1.0,
) -> dict[str, Any]:
    """Inspect one OBB without changing it.

    DOTA tiles can contain edge objects whose corners fall outside a crop.  Such
    boxes are useful for auditing, but they should not silently enter coordinate
    regression targets because the requested answer would be outside the image.
    """
    points = [[float(x), float(y)] for x, y in obj.get("points_pixel", [])]
    area = polygon_area(points) if len(points) == 4 else 0.0
    non_finite = any(not math.isfinite(v) for point in points for v in point)
    out_of_bounds = any(
        x < 0.0 or y < 0.0 or x > max(0.0, width - 1.0) or y > max(0.0, height - 1.0)
        for x, y in points
    ) if len(points) == 4 and not non_finite else True
    reasons: list[str] = []
    if len(points) != 4:
        reasons.append("not_four_points")
    if non_finite:
        reasons.append("non_finite")
    if area <= float(min_area):
        reasons.append("degenerate_area")
    if out_of_bounds:
        reasons.append("out_of_bounds")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "area": float(area),
        "out_of_bounds": bool(out_of_bounds),
        "non_finite": bool(non_finite),
    }


def sanitize_obb_object(
    obj: dict[str, Any],
    width: int,
    height: int,
    policy: str = "drop",
    min_area: float = 1.0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Apply the configured invalid-object policy.

    Policies:
    - ``drop``: exclude invalid targets from Ref, Direct SFT and evaluation.
    - ``clamp``: clamp corners to the image, then keep only a non-degenerate OBB.
    - ``fail``: abort immediately.
    """
    status = inspect_obb_object(obj, width, height, min_area)
    if status["valid"]:
        return dict(obj), {**status, "action": "keep"}

    policy = str(policy or "drop").lower()
    if policy == "fail":
        raise ValueError(
            f"Invalid OBB object_index={obj.get('object_index')} reasons={status['reasons']}"
        )
    if policy == "drop":
        return None, {**status, "action": "drop"}
    if policy != "clamp":
        raise ValueError(f"Unknown invalid_object_policy: {policy}")

    clamped = [
        [
            max(0.0, min(max(0.0, width - 1.0), float(x))),
            max(0.0, min(max(0.0, height - 1.0), float(y))),
        ]
        for x, y in obj.get("points_pixel", [])
    ]
    repaired = dict(obj)
    repaired["points_pixel"] = clamped
    repaired["center_pixel"] = [
        sum(x for x, _ in clamped) / 4.0,
        sum(y for _, y in clamped) / 4.0,
    ]
    repaired["area_pixel"] = polygon_area(clamped)
    repaired_status = inspect_obb_object(repaired, width, height, min_area)
    if repaired_status["valid"]:
        return repaired, {
            **repaired_status,
            "action": "clamp",
            "original_reasons": status["reasons"],
        }
    return None, {
        **repaired_status,
        "action": "drop_after_clamp",
        "original_reasons": status["reasons"],
    }

def norm100(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [[round(x / max(width, 1) * 100, 4), round(y / max(height, 1) * 100, 4)] for x, y in points]


def format_points(points: Iterable[Iterable[float]]) -> str:
    return ",".join(str(int(round(v))) for p in points for v in p)


def region_name(cx: float, cy: float, width: int, height: int) -> str:
    xs, ys = ["left", "center", "right"], ["upper", "middle", "lower"]
    xi = min(2, max(0, int(cx / max(width, 1) * 3)))
    yi = min(2, max(0, int(cy / max(height, 1) * 3)))
    return "central area" if (xi, yi) == (1, 1) else f"{ys[yi]}-{xs[xi]} area"


def direction(dx: float, dy: float) -> str:
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    names = ["right of", "upper-right of", "above", "upper-left of", "left of", "lower-left of", "below", "lower-right of"]
    return names[int((angle + 22.5) // 45) % 8]


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def image_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    raw = str(text or "").strip().replace("```json", "").replace("```", "")
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    for start in [m.start() for m in re.finditer(r"\{", raw)]:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{": depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(raw[start:i + 1])
                        return value if isinstance(value, dict) else None
                    except Exception:
                        break
    return None


def extract_final_text(text: str | None) -> str:
    out = str(text or "").strip()
    if "</think>" in out:
        out = out.rsplit("</think>", 1)[-1].strip()
    for prefix in ("[Final Answer]:", "FINAL_ANSWER:", "Answer:"):
        if prefix.lower() in out.lower():
            idx = out.lower().rfind(prefix.lower())
            out = out[idx + len(prefix):].strip()
    return out


def infer_and_convert_points(points: list[list[float]], width: int, height: int, mode: str = "auto") -> tuple[list[list[float]], str]:
    flat = [abs(float(v)) for p in points for v in p]
    vmax = max(flat) if flat else 0
    detected = mode
    if mode == "auto":
        if vmax <= 1.5: detected = "normalized_0_1"
        elif vmax <= 100.5 and max(width, height) > 200: detected = "normalized_0_100"
        elif vmax <= 1000.5 and max(width, height) > 1000: detected = "normalized_0_1000"
        else: detected = "pixel"
    converted = []
    for x, y in points:
        if detected == "normalized_0_1": x, y = x * width, y * height
        elif detected == "normalized_0_100": x, y = x / 100 * width, y / 100 * height
        elif detected == "normalized_0_1000": x, y = x / 1000 * width, y / 1000 * height
        converted.append([max(0.0, min(width - 1.0, x)), max(0.0, min(height - 1.0, y))])
    return converted, detected


_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_CONFIDENCE_RE = re.compile(r"(?:0|0?\.\d+|1(?:\.0+)?)")
_COORDINATE_PUNCTUATION_RE = re.compile(r"^[\s,\[\]\(\)]+$")


def _coordinates_in_declared_mode(
    points: list[list[float]],
    width: int,
    height: int,
    mode: str,
) -> bool:
    """Reject out-of-range formal predictions before conversion/clamping."""
    if mode == "auto":
        return all(math.isfinite(v) for point in points for v in point)
    limits = {
        "normalized_0_1": (1.0, 1.0),
        "normalized_0_100": (100.0, 100.0),
        "normalized_0_1000": (1000.0, 1000.0),
        "pixel": (max(0.0, float(width - 1)), max(0.0, float(height - 1))),
    }
    if mode not in limits:
        return False
    xmax, ymax = limits[mode]
    return all(
        math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= xmax and 0.0 <= y <= ymax
        for x, y in points
    )


def parse_obb_line(
    line: str,
    width: int,
    height: int,
    coord_mode: str = "auto",
    *,
    strict: bool = False,
    expected_class: str | None = None,
    require_confidence: bool = False,
) -> dict[str, Any] | None:
    """Parse one OBB answer line.

    ``strict=False`` preserves the permissive parser for legacy trajectory
    diagnostics. Formal evaluation must use ``strict=True`` so explanatory
    prose or the first eight numbers of a malformed answer cannot become a
    prediction.
    """
    cleaned = line.strip().strip("`").strip()
    parts = [x.strip() for x in cleaned.split("|")]
    if strict and len(parts) not in {2, 3}:
        return None
    if not strict and len(parts) < 2:
        return None

    cls = normalize_class_name(parts[0])
    wanted = normalize_class_name(expected_class) if expected_class is not None else None
    if cls is None or (wanted is not None and cls != wanted):
        return None

    confidence, start = 1.0, 1
    has_confidence = len(parts) >= 3 and _CONFIDENCE_RE.fullmatch(parts[1]) is not None
    if has_confidence:
        confidence, start = float(parts[1]), 2
    elif strict and (len(parts) == 3 or require_confidence):
        return None

    coordinate_field = "|".join(parts[start:])
    number_matches = list(re.finditer(_NUMBER_PATTERN, coordinate_field))
    nums = [float(match.group(0)) for match in number_matches]
    if strict:
        if len(nums) != 8:
            return None
        remainder = re.sub(_NUMBER_PATTERN, "", coordinate_field)
        if remainder and _COORDINATE_PUNCTUATION_RE.fullmatch(remainder) is None:
            return None
    elif len(nums) < 8:
        return None
    nums = nums[:8]

    pts_raw = [[nums[i], nums[i + 1]] for i in range(0, 8, 2)]
    if strict and not _coordinates_in_declared_mode(pts_raw, width, height, coord_mode):
        return None
    pts, detected = infer_and_convert_points(pts_raw, width, height, coord_mode)
    if polygon_area(pts) <= 1:
        return None
    return {
        "class_id": DOTA_CLASSES.index(cls),
        "class_name": cls,
        "confidence": confidence,
        "points": pts,
        "coordinate_mode_detected": detected,
        "source_line": cleaned,
    }


def parse_obb_output_status(
    text: str | None,
    width: int,
    height: int,
    coord_mode: str = "auto",
    *,
    strict: bool = True,
    require_markers: bool = False,
    expected_class: str | None = None,
    require_confidence: bool = False,
    exactly_one: bool = False,
) -> dict[str, Any]:
    """Parse an OBB response and retain protocol diagnostics.

    A failed formal parse always returns zero predictions. This is intentional:
    IoU must be 0 for malformed answers rather than being calculated from eight
    unrelated numbers in chain-of-thought text.
    """
    final = extract_final_text(text)
    matches = list(
        re.finditer(r"FINAL_DETECTIONS\s*(.*?)\s*END_DETECTIONS", final, re.I | re.S)
    )
    marker_ok = len(matches) == 1
    if require_markers and not marker_ok:
        return {
            "predictions": [],
            "format_ok": False,
            "parse_ok": False,
            "reason": "missing_or_duplicated_detection_markers",
            "nonempty_lines": 0,
        }

    block = matches[0].group(1) if marker_ok else final
    lines = [line.strip() for line in block.splitlines() if line.strip() and line.strip() != "```"]
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    bad_lines = 0
    for line in lines:
        item = parse_obb_line(
            line,
            width,
            height,
            coord_mode,
            strict=strict,
            expected_class=expected_class,
            require_confidence=require_confidence,
        )
        if item is None:
            bad_lines += 1
            continue
        key = (
            item["class_id"],
            tuple(round(v, 2) for point in item["points"] for v in point),
        )
        if key not in seen:
            parsed.append(item)
            seen.add(key)

    parse_ok = bad_lines == 0 and (not exactly_one or len(parsed) == 1)
    if exactly_one and not lines:
        parse_ok = False
    format_ok = (marker_ok if require_markers else True) and parse_ok
    if not parse_ok:
        reason = "wrong_prediction_count" if exactly_one and bad_lines == 0 else "malformed_prediction_line"
        parsed = []
    else:
        reason = None
    return {
        "predictions": parsed,
        "format_ok": bool(format_ok),
        "parse_ok": bool(parse_ok),
        "reason": reason,
        "nonempty_lines": len(lines),
    }


def parse_obb_output(
    text: str | None,
    width: int,
    height: int,
    coord_mode: str = "auto",
    *,
    strict: bool = False,
    require_markers: bool = False,
    expected_class: str | None = None,
    require_confidence: bool = False,
    exactly_one: bool = False,
) -> list[dict[str, Any]]:
    return parse_obb_output_status(
        text,
        width,
        height,
        coord_mode,
        strict=strict,
        require_markers=require_markers,
        expected_class=expected_class,
        require_confidence=require_confidence,
        exactly_one=exactly_one,
    )["predictions"]


def count_optimizer_steps(samples: int, epochs: float, per_device_batch: int, grad_accum: int, world_size: int) -> int:
    return math.ceil(samples / max(1, per_device_batch * grad_accum * world_size)) * math.ceil(epochs)


def hbb_from_points(points: Iterable[Iterable[float]]) -> list[float]:
    arr = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]


def points_to_norm(points: Iterable[Iterable[float]], width: int, height: int, scale: float) -> list[list[float]]:
    return [[round(float(x) / max(width, 1) * scale, 4), round(float(y) / max(height, 1) * scale, 4)] for x, y in points]


def hbb_to_norm(box: Iterable[float], width: int, height: int, scale: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [round(x1 / max(width, 1) * scale, 4), round(y1 / max(height, 1) * scale, 4), round(x2 / max(width, 1) * scale, 4), round(y2 / max(height, 1) * scale, 4)]


def hbb_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def split_center_rois(
    items: list[dict[str, Any]],
    max_objects: int,
    width: int,
    height: int,
) -> list[tuple[list[float], list[dict[str, Any]]]]:
    """Split dense object lists into disjoint center-conditioned image ROIs."""
    pending: list[tuple[list[float], list[dict[str, Any]]]] = [
        ([0.0, 0.0, float(width), float(height)], list(items))
    ]
    output: list[tuple[list[float], list[dict[str, Any]]]] = []
    while pending:
        roi, group = pending.pop()
        if len(group) <= max(1, max_objects):
            output.append((roi, group))
            continue
        x_values = [float(row["center_pixel"][0]) for row in group]
        y_values = [float(row["center_pixel"][1]) for row in group]
        axes = (
            [0, 1]
            if max(x_values) - min(x_values) >= max(y_values) - min(y_values)
            else [1, 0]
        )
        split_done = False
        for axis in axes:
            ordered = sorted(
                group,
                key=lambda row: (
                    row["center_pixel"][axis],
                    row.get("object_index", row.get("id", "")),
                ),
            )
            middle = len(ordered) // 2
            left, right = ordered[:middle], ordered[middle:]
            left_value = float(left[-1]["center_pixel"][axis])
            right_value = float(right[0]["center_pixel"][axis])
            if left_value >= right_value:
                continue
            boundary = (left_value + right_value) / 2.0
            x1, y1, x2, y2 = roi
            if axis == 0:
                pending.extend(
                    [
                        ([boundary, y1, x2, y2], right),
                        ([x1, y1, boundary, y2], left),
                    ]
                )
            else:
                pending.extend(
                    [
                        ([x1, boundary, x2, y2], right),
                        ([x1, y1, x2, boundary], left),
                    ]
                )
            split_done = True
            break
        if not split_done:
            raise RuntimeError(
                "Cannot form disjoint ROIs because too many objects share the same center."
            )
    return sorted(output, key=lambda pair: (pair[0][1], pair[0][0]))


def format_obb_answer(class_name: str, points: Iterable[Iterable[float]]) -> str:
    return f"{class_name}|" + ",".join(str(int(round(float(v)))) for p in points for v in p)


def canonical_obb_points(points: Iterable[Iterable[float]]) -> list[list[float]]:
    """Return four OBB corners in a stable clockwise order.

    Image coordinates have the y axis pointing down. Sorting by atan2 in this
    coordinate system yields clockwise traversal. The first point is chosen as
    the top-most corner, breaking ties by x, so token supervision is stable.
    """
    arr = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    if arr.shape != (4, 2):
        raise ValueError("OBB 必须包含四个点")
    c = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    ordered = arr[np.argsort(angles)]
    start = min(range(4), key=lambda i: (ordered[i, 1], ordered[i, 0]))
    ordered = np.concatenate([ordered[start:], ordered[:start]], axis=0)
    return [[float(x), float(y)] for x, y in ordered]


def clamp_points(points: Iterable[Iterable[float]], width: int, height: int) -> list[list[float]]:
    return [
        [max(0.0, min(float(width), float(x))), max(0.0, min(float(height), float(y)))]
        for x, y in points
    ]


def format_number(value: float, decimals: int = 0) -> str:
    if decimals <= 0:
        return str(int(round(float(value))))
    text = f"{float(value):.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def format_flat_points(points: Iterable[Iterable[float]], decimals: int = 0) -> str:
    return ",".join(format_number(v, decimals) for p in points for v in p)


def coordinate_points(
    points_pixel: Iterable[Iterable[float]],
    width: int,
    height: int,
    target: str,
) -> list[list[float]]:
    pts = canonical_obb_points(points_pixel)
    if target == "pixel_obb":
        return pts
    if target == "norm100_obb":
        return points_to_norm(pts, width, height, 100.0)
    if target == "norm1000_obb":
        return points_to_norm(pts, width, height, 1000.0)
    raise ValueError(f"不支持的 OBB 坐标目标: {target}")


def coordinate_instruction(target: str, width: int, height: int) -> str:
    if target == "pixel_obb":
        return f"original-image pixel coordinates for an image of size {width}x{height}"
    if target == "norm100_obb":
        return "normalized coordinates in [0,100] relative to the original image"
    if target == "norm1000_obb":
        return "normalized coordinates in [0,1000] relative to the original image"
    raise ValueError(f"未知坐标制: {target}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_and_convert_hbb(box: Iterable[float], width: int, height: int, mode: str = "auto") -> tuple[list[float], str]:
    values = [float(v) for v in box]
    if len(values) != 4:
        raise ValueError("HBB requires four values")
    vmax = max(abs(v) for v in values)
    detected = mode
    if mode == "auto":
        if vmax <= 1.5:
            detected = "normalized_0_1"
        elif vmax <= 100.5:
            detected = "normalized_0_100"
        elif vmax <= 1000.5:
            detected = "normalized_0_1000"
        else:
            detected = "pixel"
    x1, y1, x2, y2 = values
    if detected == "normalized_0_1":
        x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
    elif detected == "normalized_0_100":
        x1, x2, y1, y2 = x1 / 100 * width, x2 / 100 * width, y1 / 100 * height, y2 / 100 * height
    elif detected == "normalized_0_1000":
        x1, x2, y1, y2 = x1 / 1000 * width, x2 / 1000 * width, y1 / 1000 * height, y2 / 1000 * height
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    return [x1, y1, x2, y2], detected


def parse_hbb_output(text: str | None, width: int, height: int, mode: str = "auto") -> dict[str, Any] | None:
    raw = extract_final_text(text)
    class_name = None
    obj = extract_json_object(raw)
    values = None
    if isinstance(obj, dict):
        values = obj.get("bbox") or obj.get("box")
        class_name = normalize_class_name(obj.get("class_name") or obj.get("class"))
    if not (isinstance(values, (list, tuple)) and len(values) == 4):
        match = re.search(r"(?:bbox|box)?\s*[:=]?\s*[\[<(]\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*[\])>]", raw, re.I)
        if match:
            values = [float(match.group(i)) for i in range(1, 5)]
    if not (isinstance(values, (list, tuple)) and len(values) == 4):
        return None
    box, detected = infer_and_convert_hbb(values, width, height, mode)
    if (box[2] - box[0]) * (box[3] - box[1]) <= 1:
        return None
    if class_name is None:
        prefix = raw.split("|", 1)[0].strip()
        class_name = normalize_class_name(prefix)
    return {"class_name": class_name, "bbox": box, "coordinate_mode_detected": detected}
