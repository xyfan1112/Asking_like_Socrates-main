"""Deterministic semantic resolver for rule-generated DOTA references.

String uniqueness is insufficient: two different phrases can still identify
the same candidates, and wording such as "the small vehicle" is ambiguous when
several small vehicles exist.  The builder stores the structured constraints
that were actually verbalized, and both the builder and validator resolve those
constraints against every valid object in the same image.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def relation_from_delta(dx: float, dy: float) -> str:
    """Describe target position relative to an anchor.

    ``dx``/``dy`` are anchor minus target.  Positive ``dx`` therefore means the
    target is left of the anchor.
    """
    if abs(dx) >= 1.5 * abs(dy):
        return "left of" if dx > 0 else "right of"
    if abs(dy) >= 1.5 * abs(dx):
        return "above" if dy > 0 else "below"
    horizontal = "left" if dx > 0 else "right"
    vertical = "above" if dy > 0 else "below"
    return f"{vertical} and {horizontal} of"


def _center(item: dict[str, Any]) -> tuple[float, float] | None:
    value = item.get("center_pixel")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _descriptor_index(item: dict[str, Any]) -> int | None:
    try:
        return int(item.get("object_index"))
    except (TypeError, ValueError):
        return None


def resolve_reference_matches(
    constraints: dict[str, Any],
    descriptors: Iterable[dict[str, Any]],
    width: int,
    height: int,
) -> list[int]:
    """Return object indices satisfying every constraint expressed in the text."""
    items = list(descriptors)
    expected_class = str(constraints.get("class_name") or "")
    expected_region = str(constraints.get("region") or "")
    expected_geometry = str(constraints.get("geometry_phrase") or "")
    expected_extreme = str(constraints.get("extreme") or "")
    anchor = constraints.get("anchor") if isinstance(constraints.get("anchor"), dict) else None
    diagonal = max(1.0, math.hypot(float(width), float(height)))

    matches: list[int] = []
    for item in items:
        object_index = _descriptor_index(item)
        center = _center(item)
        if object_index is None or center is None:
            continue
        if str(item.get("class_name") or "") != expected_class:
            continue
        if expected_region and str(item.get("semantic_region") or "") != expected_region:
            continue
        if (
            expected_geometry
            and str(item.get("geometry_phrase_used") or "") != expected_geometry
        ):
            continue
        extremes = {
            str(value)
            for value in (item.get("semantic_extremes") or [])
            if str(value)
        }
        if expected_extreme and expected_extreme not in extremes:
            continue

        if anchor:
            anchor_class = str(anchor.get("class_name") or "")
            anchor_items = [
                candidate
                for candidate in items
                if str(candidate.get("class_name") or "") == anchor_class
                and _descriptor_index(candidate) != object_index
                and _center(candidate) is not None
            ]
            if bool(anchor.get("require_unique_class", True)) and len(anchor_items) != 1:
                continue
            if not anchor_items:
                continue
            relation_ok = False
            max_distance_ratio = float(anchor.get("max_distance_ratio", 0.35))
            for anchor_item in anchor_items:
                anchor_center = _center(anchor_item)
                assert anchor_center is not None
                dx = anchor_center[0] - center[0]
                dy = anchor_center[1] - center[1]
                if math.hypot(dx, dy) > max_distance_ratio * diagonal:
                    continue
                if relation_from_delta(dx, dy) == str(anchor.get("relation") or ""):
                    relation_ok = True
                    break
            if not relation_ok:
                continue

        matches.append(object_index)
    return sorted(matches)


def semantic_unique_for_target(
    constraints: dict[str, Any],
    descriptors: Iterable[dict[str, Any]],
    width: int,
    height: int,
    target_object_index: int,
) -> tuple[bool, list[int]]:
    matches = resolve_reference_matches(constraints, descriptors, width, height)
    return matches == [int(target_object_index)], matches
