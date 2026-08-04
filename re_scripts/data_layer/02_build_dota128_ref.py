#!/usr/bin/env python3
"""Build DOTA-Ref v4.3.3 from the original DOTA OBB supervision.

Key rules:
- keep every valid OBB in <split>_all.jsonl for Direct supervision;
- select references whose structured semantics resolve to exactly one object;
- separately mark a conservative Socratic-friendly subset;
- never use ordinal wording or phrases such as "at the bottom of its class";
- omit unstable orientation adjectives for symmetric/region classes;
- use stable different-class anchors only;
- identify every Socratic target with an automatically derived coarse focus box;
- keep classification references masked from the canonical label;
- change row IDs whenever the query schema changes, preventing unsafe resume.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main_layer"))
from common import (  # noqa: E402
    CUSTOM_TAXONOMY,
    DOTA_CLASSES,
    QA_LANGUAGE,
    TAXONOMY_SHA256,
    canonical_obb_points,
    coordinate_instruction,
    coordinate_points,
    discover_images,
    format_flat_points,
    hbb_from_points,
    hbb_to_norm,
    load_obb_label,
    load_settings,
    points_to_norm,
    order_polygon,
    region_name,
    sanitize_obb_object,
    settings_from_cli,
    stable_id,
    write_json,
    write_jsonl,
)
from data_layer.ref_semantics import (  # noqa: E402
    relation_from_delta,
    semantic_unique_for_target,
)
from data_layer.qa_i18n import (  # noqa: E402
    classification_question as render_classification_question,
    grounding_question as render_grounding_question,
    render_reference,
)

SCHEMA_VERSION = "dota_ref_v4_3_3"
ID_SALT = "dota_ref_v4_3_3_semantic_unique_perceiver_roi"

# Orientation is visually unstable or semantically unhelpful for these classes.
NO_ORIENTATION_CLASSES = {"storage tank", "roundabout", "harbor"}
NO_ASPECT_CLASSES = {"harbor"}
REGION_CLASSES = {"harbor"}

# The canonical class is hidden, but the phrase remains visually meaningful.
# These are deliberately broader than the v4.2 labels (for example, no direct
# "road vehicle" or "aerial object" shortcut to a tiny answer subset).
CLASS_FAMILY = {
    "plane": "airfield target",
    "helicopter": "airfield target",
    "ship": "water-associated target",
    "harbor": "waterfront region",
    "storage tank": "circular industrial structure",
    "roundabout": "circular transport structure",
    "bridge": "linear crossing structure",
    "large vehicle": "road-side target",
    "small vehicle": "road-side target",
    "baseball diamond": "marked recreational facility",
    "tennis court": "marked recreational facility",
    "basketball court": "marked recreational facility",
    "ground track field": "marked recreational facility",
    "soccer ball field": "marked recreational facility",
    "swimming pool": "recreational water facility",
}

GROUND_HEAD = {
    "harbor": "harbor region",
}

# Custom taxonomies have no assumed hierarchy, aliases, region classes, or
# DOTA-specific visual ontology.  Geometry remains available for every class.
if CUSTOM_TAXONOMY:
    NO_ORIENTATION_CLASSES = set()
    NO_ASPECT_CLASSES = set()
    REGION_CLASSES = set()
    CLASS_FAMILY = {}
    GROUND_HEAD = {}

EXTREME_WORD = {
    "leftmost": "leftmost",
    "rightmost": "rightmost",
    "topmost": "highest",
    "bottommost": "lowest",
}


def _geometry(obj: dict[str, Any]) -> dict[str, Any]:
    pts = order_polygon(obj["points_pixel"]).astype(np.float64)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    long_side = float(lengths[long_index])
    short_side = max(1e-6, float(np.min(lengths)))
    aspect = float(long_side / short_side)
    dx, dy = edges[long_index]
    angle = math.degrees(math.atan2(float(dy), float(dx)))
    angle %= 180.0
    if angle < 22.5 or angle >= 157.5:
        orientation = "roughly horizontal"
    elif 67.5 <= angle < 112.5:
        orientation = "roughly vertical"
    elif angle < 67.5:
        orientation = "diagonally descending to the right"
    else:
        orientation = "diagonally ascending to the right"
    if aspect >= 3.5:
        shape = "strongly elongated"
    elif aspect >= 1.8:
        shape = "elongated"
    elif aspect <= 1.25:
        shape = "compact"
    else:
        shape = "moderately rectangular"
    return {
        "aspect_ratio": round(aspect, 4),
        "orientation_degrees": round(angle, 3),
        "orientation_phrase": orientation,
        "shape_phrase": shape,
    }


def _geometry_phrase(class_name: str, geom: dict[str, Any]) -> str:
    parts: list[str] = []
    if class_name not in NO_ASPECT_CLASSES:
        if class_name in {"storage tank", "roundabout"}:
            parts.append("compact")
        elif geom["aspect_ratio"] >= 1.8:
            parts.append(geom["shape_phrase"])
    if class_name not in NO_ORIENTATION_CLASSES:
        parts.append(geom["orientation_phrase"])
    return " and ".join(parts)


def _unique_extreme(obj: dict[str, Any], same: list[dict[str, Any]], axis: str, tolerance: float) -> bool:
    if len(same) < 2:
        return False
    idx = 0 if axis in {"leftmost", "rightmost"} else 1
    values = [float(o["center_pixel"][idx]) for o in same]
    value = float(obj["center_pixel"][idx])
    target = min(values) if axis in {"leftmost", "topmost"} else max(values)
    if abs(value - target) > 1e-6:
        return False
    ordered = sorted(values) if axis in {"leftmost", "topmost"} else sorted(values, reverse=True)
    return abs(ordered[0] - ordered[1]) >= tolerance


def _relation(dx: float, dy: float) -> str:
    return relation_from_delta(dx, dy)


def _stable_different_class_anchor(
    obj: dict[str, Any], objects: list[dict[str, Any]], width: int, height: int
) -> dict[str, Any] | None:
    """Choose a visually stable, different-class anchor.

    Same-class "nearest" relations are intentionally disabled because dense DOTA
    scenes often contain several nearly equidistant ships/vehicles.  Because the
    generated text says "the <anchor class>" without encoding a nearest rule, an
    anchor class is usable only when exactly one such object exists in the image.
    """
    # With arbitrary flat labels there is no safe non-leaking family name for an
    # anchor. Disable anchor wording rather than inventing a hierarchy.
    if CUSTOM_TAXONOMY:
        return None
    cx, cy = obj["center_pixel"]
    diag = max(1e-6, math.hypot(width, height))
    candidates: list[tuple[float, dict[str, Any], float, float]] = []
    for other in objects:
        if other["object_index"] == obj["object_index"]:
            continue
        if other["class_name"] == obj["class_name"]:
            continue
        ox, oy = other["center_pixel"]
        distance = math.hypot(ox - cx, oy - cy)
        if distance <= 0.35 * diag:
            candidates.append((distance, other, ox - cx, oy - cy))
    if not candidates:
        return None

    by_anchor_class: dict[str, list[tuple[float, dict[str, Any], float, float]]] = defaultdict(list)
    for candidate in candidates:
        by_anchor_class[candidate[1]["class_name"]].append(candidate)
    global_anchor_counts = Counter(
        other["class_name"]
        for other in objects
        if other["object_index"] != obj["object_index"]
        and other["class_name"] != obj["class_name"]
    )
    stable: list[tuple[float, dict[str, Any], float, float]] = []
    for anchor_class, items in by_anchor_class.items():
        items.sort(key=lambda x: x[0])
        if len(items) == 1 and global_anchor_counts[anchor_class] == 1:
            stable.append(items[0])
    if not stable:
        return None
    distance, other, dx, dy = min(stable, key=lambda x: x[0])
    return {
        "class_name": other["class_name"],
        "same_class": False,
        "distance_ratio": round(distance / diag, 4),
        "relation": _relation(dx, dy),
        "object_index": other["object_index"],
        "stability_rule": "different_class_globally_unique",
        "require_unique_class": True,
    }


def _class_family(class_name: str) -> str:
    if CUSTOM_TAXONOMY:
        return "target object"
    return CLASS_FAMILY.get(class_name, "target")


def _head(class_name: str, masked: bool) -> str:
    return _class_family(class_name) if masked else GROUND_HEAD.get(class_name, class_name)


def _base_phrase(class_name: str, geom_phrase: str, region: str, masked: bool) -> str:
    head = _head(class_name, masked)
    descriptors = f"{geom_phrase} " if geom_phrase else ""
    return f"the {descriptors}{head} in the {region}"


def _extreme_phrase(
    class_name: str, geom_phrase: str, region: str, axis: str, masked: bool
) -> str:
    head = _head(class_name, masked)
    descriptors = f"{geom_phrase} " if geom_phrase else ""
    word = EXTREME_WORD[axis]
    group = (
        "visually similar targets"
        if masked
        else f"all {GROUND_HEAD.get(class_name, class_name)} instances"
    )
    return (
        f"the {word} {descriptors}{head} among {group} in the image, "
        f"located in the {region}"
    )


def _anchor_phrase(
    class_name: str,
    geom_phrase: str,
    region: str,
    anchor: dict[str, Any],
    masked: bool,
) -> str:
    base = _base_phrase(class_name, geom_phrase, region, masked)
    anchor_head = _class_family(anchor["class_name"]) if masked else anchor["class_name"]
    return f"{base}, {anchor['relation']} the {anchor_head}"


def _build_candidate(
    obj: dict[str, Any], objects: list[dict[str, Any]], width: int, height: int
) -> dict[str, Any]:
    same = [x for x in objects if x["class_name"] == obj["class_name"]]
    geom = _geometry(obj)
    cx, cy = obj["center_pixel"]
    region = region_name(cx, cy, width, height)
    tolerance = max(6.0, 0.025 * min(width, height))
    geom_phrase = _geometry_phrase(obj["class_name"], geom)

    variants: list[dict[str, Any]] = [
        {
            "grounding": _base_phrase(obj["class_name"], geom_phrase, region, False),
            "classification": _base_phrase(obj["class_name"], geom_phrase, region, True),
            "score": 1.0,
            "features": ["region"] + (["geometry"] if geom_phrase else []),
            "semantic_constraints": {
                "class_name": obj["class_name"],
                "region": region,
                "geometry_phrase": geom_phrase,
                "extreme": None,
                "anchor": None,
            },
        }
    ]

    extremes: list[str] = []
    for axis in ("leftmost", "rightmost", "topmost", "bottommost"):
        if _unique_extreme(obj, same, axis, tolerance):
            extremes.append(axis)
            variants.append(
                {
                    "grounding": _extreme_phrase(obj["class_name"], geom_phrase, region, axis, False),
                    "classification": _extreme_phrase(obj["class_name"], geom_phrase, region, axis, True),
                    "score": 4.0,
                    "features": ["region", f"extreme:{axis}"] + (["geometry"] if geom_phrase else []),
                    "semantic_constraints": {
                        "class_name": obj["class_name"],
                        "region": region,
                        "geometry_phrase": geom_phrase,
                        "extreme": axis,
                        "anchor": None,
                    },
                }
            )

    anchor = _stable_different_class_anchor(obj, objects, width, height)
    if anchor:
        variants.append(
            {
                "grounding": _anchor_phrase(obj["class_name"], geom_phrase, region, anchor, False),
                "classification": _anchor_phrase(obj["class_name"], geom_phrase, region, anchor, True),
                "score": 3.6,
                "features": ["region", "different_class_anchor"] + (["geometry"] if geom_phrase else []),
                "semantic_constraints": {
                    "class_name": obj["class_name"],
                    "region": region,
                    "geometry_phrase": geom_phrase,
                    "extreme": None,
                    "anchor": {
                        "class_name": anchor["class_name"],
                        "relation": anchor["relation"],
                        "require_unique_class": True,
                        "max_distance_ratio": 0.35,
                    },
                },
            }
        )
        if extremes:
            axis = extremes[0]
            ground = _extreme_phrase(obj["class_name"], geom_phrase, region, axis, False)
            masked = _extreme_phrase(obj["class_name"], geom_phrase, region, axis, True)
            masked_anchor = _class_family(anchor["class_name"])
            variants.append(
                {
                    "grounding": f"{ground}, {anchor['relation']} the {anchor['class_name']}",
                    "classification": f"{masked}, {anchor['relation']} the {masked_anchor}",
                    "score": 6.6,
                    "features": ["region", f"extreme:{axis}", "different_class_anchor"] + (["geometry"] if geom_phrase else []),
                    "semantic_constraints": {
                        "class_name": obj["class_name"],
                        "region": region,
                        "geometry_phrase": geom_phrase,
                        "extreme": axis,
                        "anchor": {
                            "class_name": anchor["class_name"],
                            "relation": anchor["relation"],
                            "require_unique_class": True,
                            "max_distance_ratio": 0.35,
                        },
                    },
                }
            )

    return {
        "region": region,
        "geometry": geom,
        "geometry_phrase_used": geom_phrase,
        "anchor": anchor,
        "variants": variants,
        "same_class_count": len(same),
        "extremes": extremes,
    }


def _semantic_descriptors(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "object_index": item["object"]["object_index"],
            "class_name": item["object"]["class_name"],
            "center_pixel": item["object"]["center_pixel"],
            "semantic_region": item["reference_candidate"]["region"],
            "geometry_phrase_used": item["reference_candidate"]["geometry_phrase_used"],
            "semantic_extremes": list(item["reference_candidate"]["extremes"]),
        }
        for item in candidates
    ]


def _choose_unique_references(
    candidates: list[dict[str, Any]],
    width: int,
    height: int,
) -> None:
    descriptors = _semantic_descriptors(candidates)
    for item in candidates:
        target_index = int(item["object"]["object_index"])
        for variant in item["reference_candidate"]["variants"]:
            unique, matches = semantic_unique_for_target(
                variant["semantic_constraints"],
                descriptors,
                width,
                height,
                target_index,
            )
            variant["semantic_unique"] = bool(unique)
            variant["semantic_match_object_indices"] = matches
        variants = sorted(
            item["reference_candidate"]["variants"],
            key=lambda v: (-v["score"], -len(v["features"]), len(v["grounding"])),
        )
        chosen = next(
            (
                v
                for v in variants
                if v["semantic_unique"]
            ),
            None,
        )
        item["reference_selected"] = chosen
        item["ref_eligible"] = chosen is not None
        item["reference_reject_reason"] = (
            None if chosen else "no_semantically_unique_non_ordinal_reference"
        )


def _focus_hbb(
    hbb_pixel: list[float],
    width: int,
    height: int,
    expansion: float,
    min_span_norm1000: float,
) -> tuple[list[float], list[float]]:
    """Build a coarse, automatically derived classification focus region.

    The region is deliberately wider than the target OBB. It identifies which
    object the classification question refers to without exposing its class or
    replacing the localization target.
    """
    x1, y1, x2, y2 = [float(v) for v in hbb_pixel]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    min_w = float(min_span_norm1000) / 1000.0 * width
    min_h = float(min_span_norm1000) / 1000.0 * height
    span_w = max((x2 - x1) * expansion, min_w)
    span_h = max((y2 - y1) * expansion, min_h)
    focus = [
        max(0.0, cx - span_w / 2.0),
        max(0.0, cy - span_h / 2.0),
        min(float(width - 1), cx + span_w / 2.0),
        min(float(height - 1), cy + span_h / 2.0),
    ]
    return focus, hbb_to_norm(focus, width, height, 1000.0)


def _socratic_quality(
    obj: dict[str, Any],
    candidate: dict[str, Any],
    width: int,
    height: int,
    min_short_side_norm1000: float,
    min_area_ratio: float,
) -> dict[str, Any]:
    points = np.asarray(obj["points_pixel"], dtype=np.float32)
    points = order_polygon(points).astype(np.float64)
    short_side = float(
        np.min(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))
    )
    short_side_norm1000 = short_side / max(1.0, float(min(width, height))) * 1000.0
    area_ratio = float(obj["area_pixel"]) / max(1.0, float(width * height))
    selected = candidate.get("reference_selected") or {}
    features = set(selected.get("features", []))
    same_class_count = int(candidate["reference_candidate"]["same_class_count"])
    strong_reference = (
        same_class_count == 1
        or any(feature.startswith("extreme:") for feature in features)
        or "different_class_anchor" in features
    )
    reasons: list[str] = []
    if selected is None or not selected:
        reasons.append("no_unique_reference")
    elif not bool(selected.get("semantic_unique")):
        reasons.append("semantic_reference_not_unique")
    if short_side_norm1000 < min_short_side_norm1000:
        reasons.append("target_too_small")
    if area_ratio < min_area_ratio:
        reasons.append("target_area_too_small")
    if not strong_reference:
        reasons.append("weak_reference_for_socratic")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "short_side_pixel": round(short_side, 4),
        "short_side_norm1000": round(short_side_norm1000, 4),
        "area_ratio": round(area_ratio, 8),
        "strong_reference": bool(strong_reference),
    }


def _select_ref_subset(rows: list[dict[str, Any]], max_per_class: int) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["ref_eligible"]:
            by_class[row["class_name"]].append(row)
    selected: list[dict[str, Any]] = []
    for items in by_class.values():
        items.sort(
            key=lambda r: (
                -float(r["reference_quality"]["score"]),
                len(r["reference_grounding"]),
                r["object_index"],
            )
        )
        selected.extend(items[:max_per_class])
    return sorted(selected, key=lambda r: r["object_index"])


def _make_row(
    split: str,
    image_path: Path,
    width: int,
    height: int,
    obj: dict[str, Any],
    coord_mode: str,
    candidate: dict[str, Any],
    coordinate_target: str,
    focus_expansion: float,
    focus_min_span_norm1000: float,
    socratic_min_short_side_norm1000: float,
    socratic_min_area_ratio: float,
) -> dict[str, Any]:
    pts_pixel = canonical_obb_points(obj["points_pixel"])
    pts_norm100 = points_to_norm(pts_pixel, width, height, 100.0)
    pts_norm1000 = points_to_norm(pts_pixel, width, height, 1000.0)
    hbb_pixel = hbb_from_points(pts_pixel)
    hbb_norm1000 = hbb_to_norm(hbb_pixel, width, height, 1000.0)
    focus_hbb_pixel, focus_hbb_norm1000 = _focus_hbb(
        hbb_pixel,
        width,
        height,
        focus_expansion,
        focus_min_span_norm1000,
    )
    selected = candidate.get("reference_selected")
    target_points = coordinate_points(pts_pixel, width, height, coordinate_target)
    target_answer = f"{obj['class_name']}|{format_flat_points(target_points, 0)}"
    instruction = coordinate_instruction(coordinate_target, width, height)

    grounding_ref = selected["grounding"] if selected else ""
    classification_ref = selected["classification"] if selected else ""
    if selected and (CUSTOM_TAXONOMY or QA_LANGUAGE != "en"):
        constraints = dict(selected.get("semantic_constraints") or {})
        grounding_ref = render_reference(
            constraints=constraints,
            class_name=obj["class_name"],
            lang=QA_LANGUAGE,
            masked=False,
        )
        classification_ref = render_reference(
            constraints=constraints,
            class_name=obj["class_name"],
            lang=QA_LANGUAGE,
            masked=True,
        )

    focus_text = ",".join(
        str(int(round(value))) for value in focus_hbb_norm1000
    )
    if selected and (CUSTOM_TAXONOMY or QA_LANGUAGE != "en"):
        grounding_question = render_grounding_question(
            width=width,
            height=height,
            focus_text=focus_text,
            reference=grounding_ref,
            coordinate_target=coordinate_target,
            lang=QA_LANGUAGE,
        )
        classification_question = render_classification_question(
            focus_text=focus_text,
            reference=classification_ref,
            class_names=DOTA_CLASSES,
            lang=QA_LANGUAGE,
        )
    else:
        grounding_question = (
            f"Image size: {width}x{height}. The target is inside the coarse focus region "
            f"[{focus_text}] in normalized [0,1000] coordinates; this region is only a "
            f"search hint, so resolve the exact target from the reference. Locate {grounding_ref}. "
            f"Use {instruction}. Return exactly one line as "
            "class_name|x1,y1,x2,y2,x3,y3,x4,y4 with four corners in clockwise order."
            if selected
            else ""
        )
        classification_question = (
            "The target is centered inside the coarse focus region "
            f"[{focus_text}] in normalized [0,1000] "
            f"coordinates and is described as {classification_ref}. "
            f"Which canonical DOTA category is it? Choose exactly one label from: {', '.join(DOTA_CLASSES)}. "
            "Use visual evidence; the focus region is only an object pointer. Return only the canonical class name."
            if selected
            else ""
        )
    socratic_quality = _socratic_quality(
        obj,
        candidate,
        width,
        height,
        socratic_min_short_side_norm1000,
        socratic_min_area_ratio,
    )

    row_salt = (
        ID_SALT
        if not CUSTOM_TAXONOMY and QA_LANGUAGE == "en"
        else f"{ID_SALT}|{QA_LANGUAGE}|{TAXONOMY_SHA256}"
    )
    row_id = stable_id(split, image_path.name, obj["object_index"], row_salt)
    return {
        "id": row_id,
        "split": split,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_width": width,
        "image_height": height,
        "scene_id": image_path.stem.split("__", 1)[0],
        "source": "dota128",
        "schema_version": SCHEMA_VERSION,
        "query_style_version": "natural_ref_focus_v4_semantic_unique_all_tasks",
        "qa_language": QA_LANGUAGE,
        "taxonomy_mode": "custom" if CUSTOM_TAXONOMY else "dota",
        "taxonomy_sha256": TAXONOMY_SHA256,
        "class_id": obj["class_id"],
        "class_name": obj["class_name"],
        "object_index": obj["object_index"],
        "label_coordinate_mode": coord_mode,
        "center_pixel": [round(float(x), 4) for x in obj["center_pixel"]],
        "obb_pixel": pts_pixel,
        "obb_norm100": pts_norm100,
        "obb_norm1000": pts_norm1000,
        "hbb_pixel": [round(float(x), 4) for x in hbb_pixel],
        "hbb_norm1000": hbb_norm1000,
        "classification_focus_hbb_pixel": [round(float(x), 4) for x in focus_hbb_pixel],
        "classification_focus_hbb_norm1000": focus_hbb_norm1000,
        "obj_corner_pixel": pts_pixel,
        "obj_corner_norm100": pts_norm100,
        "reference_grounding": grounding_ref,
        "reference_classification": classification_ref,
        "reference": grounding_ref,
        "grounding_question": grounding_question,
        "classification_question": classification_question,
        "question": grounding_question,
        "grounding_answer": target_answer,
        "classification_answer": obj["class_name"],
        "answer": target_answer,
        "coordinate_target": coordinate_target,
        "ref_eligible": bool(candidate.get("ref_eligible")),
        "socratic_eligible": bool(socratic_quality["eligible"]),
        "socratic_reject_reasons": list(socratic_quality["reasons"]),
        "visibility": {
            key: value
            for key, value in socratic_quality.items()
            if key not in {"eligible", "reasons", "strong_reference"}
        },
        "reference_reject_reason": candidate.get("reference_reject_reason"),
        "reference_quality": {
            "score": float(selected["score"]) if selected else 0.0,
            "features": list(selected["features"]) if selected else [],
            "unique_non_ordinal": bool(selected),
            "same_class_count": int(candidate["reference_candidate"]["same_class_count"]),
            "strong_for_socratic": bool(socratic_quality["strong_reference"]),
        },
        "reference_region": candidate["reference_candidate"]["region"],
        "semantic_extremes": list(candidate["reference_candidate"]["extremes"]),
        "reference_semantics": {
            "constraints": dict(selected["semantic_constraints"]) if selected else {},
            "match_object_indices": list(
                selected.get("semantic_match_object_indices", [])
            )
            if selected
            else [],
            "semantic_unique": bool(selected and selected.get("semantic_unique")),
        },
        "geometry": candidate["reference_candidate"]["geometry"],
        "geometry_phrase_used": candidate["reference_candidate"]["geometry_phrase_used"],
        "anchor": candidate["reference_candidate"]["anchor"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    src = Path(settings["paths"]["dota128_root"])
    out = Path(settings["paths"]["dota128_ref_root"])
    out.mkdir(parents=True, exist_ok=True)
    cfg = settings["data_conversion"]
    coordinate_target = cfg.get("coordinate_target", "norm1000_obb")
    max_per_class = int(cfg.get("max_refs_per_class_per_image", 3))
    focus_expansion = float(cfg.get("classification_focus_expansion", 2.5))
    focus_min_span_norm1000 = float(cfg.get("classification_focus_min_span_norm1000", 80.0))
    socratic_min_short_side_norm1000 = float(
        cfg.get("socratic_min_short_side_norm1000", 20.0)
    )
    socratic_min_area_ratio = float(cfg.get("socratic_min_area_ratio", 0.0002))
    quality = settings.get("data_quality", {})
    invalid_policy = str(quality.get("invalid_object_policy", "drop")).lower()
    min_area = float(quality.get("min_polygon_area", 1.0))

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "query_style_version": "natural_ref_focus_v4_semantic_unique_all_tasks",
        "coordinate_target": coordinate_target,
        "qa_language": QA_LANGUAGE,
        "taxonomy_mode": "custom" if CUSTOM_TAXONOMY else "dota",
        "taxonomy_sha256": TAXONOMY_SHA256,
        "num_classes": len(DOTA_CLASSES),
        "policy": {
            "raw_source_unchanged": True,
            "all_valid_objects_preserved": True,
            "unique_ref_only": bool(cfg.get("require_unique_reference", True)),
            "ordinal_fallback": False,
            "max_refs_per_class_per_image": max_per_class,
            "invalid_object_policy": invalid_policy,
            "min_polygon_area": min_area,
            "same_class_anchor_enabled": False,
            "stable_different_class_anchor_only": True,
            "anchor_class_must_be_globally_unique": True,
            "semantic_reference_resolver": "structured_constraints_all_objects_same_image",
            "orientation_omitted_for": sorted(NO_ORIENTATION_CLASSES),
            "classification_focus_expansion": focus_expansion,
            "classification_focus_min_span_norm1000": focus_min_span_norm1000,
            "socratic_min_short_side_norm1000": socratic_min_short_side_norm1000,
            "socratic_min_area_ratio": socratic_min_area_ratio,
        },
        "splits": {},
    }

    for split in cfg.get("splits", ["train", "val"]):
        discovered = discover_images(src, split)
        if args.limit_images:
            discovered = discovered[: args.limit_images]
        all_rows: list[dict[str, Any]] = []
        ref_rows: list[dict[str, Any]] = []
        raw_object_count = 0
        dropped_invalid_count = 0
        repaired_count = 0
        rejected: Counter[str] = Counter()
        class_all: Counter[str] = Counter()
        class_ref: Counter[str] = Counter()
        socratic_rejected: Counter[str] = Counter()

        for image_path, label_path in discovered:
            with Image.open(image_path) as image:
                width, height = image.size
            raw_objects, coord_mode = load_obb_label(label_path, width, height)
            raw_object_count += len(raw_objects)
            objects: list[dict[str, Any]] = []
            for obj in raw_objects:
                sanitized, status = sanitize_obb_object(
                    obj, width, height, policy=invalid_policy, min_area=min_area
                )
                if sanitized is None:
                    dropped_invalid_count += 1
                    reason = "+".join(status.get("reasons", [])) or status.get("action", "drop")
                    rejected[f"invalid:{reason}"] += 1
                    continue
                if status.get("action") == "clamp":
                    repaired_count += 1
                    rejected["repaired_by_clamp"] += 1
                objects.append(sanitized)

            candidates = [
                {"object": obj, "reference_candidate": _build_candidate(obj, objects, width, height)}
                for obj in objects
            ]
            _choose_unique_references(candidates, width, height)
            made = [
                _make_row(
                    split,
                    image_path,
                    width,
                    height,
                    item["object"],
                    coord_mode,
                    item,
                    coordinate_target,
                    focus_expansion,
                    focus_min_span_norm1000,
                    socratic_min_short_side_norm1000,
                    socratic_min_area_ratio,
                )
                for item in candidates
            ]
            selected = _select_ref_subset(made, max_per_class)
            selected_ids = {row["id"] for row in selected}
            for row in made:
                class_all[row["class_name"]] += 1
                if row["id"] in selected_ids:
                    row["selected_for_ref"] = True
                    ref_rows.append(row)
                    class_ref[row["class_name"]] += 1
                    for reason in row["socratic_reject_reasons"]:
                        socratic_rejected[reason] += 1
                else:
                    row["selected_for_ref"] = False
                    if row["ref_eligible"]:
                        row["reference_reject_reason"] = "class_image_ref_quota"
                    rejected[row["reference_reject_reason"] or "not_selected"] += 1
                all_rows.append(row)

        all_path = out / f"{split}_all.jsonl"
        ref_path = out / f"{split}.jsonl"
        socratic_path = out / f"{split}_socratic.jsonl"
        socratic_rows = [row for row in ref_rows if row["socratic_eligible"]]
        write_jsonl(all_path, all_rows)
        write_jsonl(ref_path, ref_rows)
        write_jsonl(socratic_path, socratic_rows)
        report["splits"][split] = {
            "images": len(discovered),
            "objects_raw": raw_object_count,
            "objects_valid": len(all_rows),
            "objects_dropped_invalid": dropped_invalid_count,
            "objects_repaired": repaired_count,
            "references_selected": len(ref_rows),
            "socratic_references_selected": len(socratic_rows),
            "reference_selection_rate": round(len(ref_rows) / max(1, len(all_rows)), 6),
            "all_file": str(all_path),
            "ref_file": str(ref_path),
            "socratic_ref_file": str(socratic_path),
            "classes_all": dict(sorted(class_all.items())),
            "classes_ref": dict(sorted(class_ref.items())),
            "rejected": dict(sorted(rejected.items())),
            "socratic_rejected": dict(sorted(socratic_rejected.items())),
        }

    report["passed"] = all(
        item.get("objects_valid", 0) > 0 and item.get("references_selected", 0) > 0
        for item in report["splits"].values()
    )
    report["next_step"] = (
        "python data_layer/03_validate_dota128_ref.py"
        if report["passed"]
        else "inspect build_report.json"
    )
    report_path = out / "build_report.json"
    write_json(report_path, report)
    print(f"[DOTA-REF BUILD] {'PASS' if report['passed'] else 'FAIL'}")
    for split, item in report["splits"].items():
        print(
            f"  {split}: raw={item['objects_raw']} valid={item['objects_valid']} "
            f"dropped={item['objects_dropped_invalid']} refs={item['references_selected']} "
            f"socratic={item['socratic_references_selected']} "
            f"selection_rate={item['reference_selection_rate']:.2%}"
        )
        print(f"    all: {item['all_file']}")
        print(f"    ref: {item['ref_file']}")
    print(f"  schema: {SCHEMA_VERSION}")
    print(f"  report: {report_path}")
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
