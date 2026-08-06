#!/usr/bin/env python3
"""Audit actual LoRA adapter tensor keys without loading the base model.

The generated adapter checkpoint is the most reliable lightweight evidence of
which module names received LoRA weights.  This script classifies tensor keys
into visual-encoder, multimodal-projector/merger, and language/other groups. It
fails closed when the settings require a frozen vision tower but obvious
visual-encoder LoRA tensors are present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

VISION_MARKERS = (
    ".visual.blocks.",
    ".visual.patch_embed.",
    ".vision_tower.",
    ".vision_model.encoder.",
    ".vision_model.embeddings.",
    ".vision_encoder.",
)
PROJECTOR_MARKERS = (
    ".visual.merger.",
    ".multi_modal_projector.",
    ".mm_projector.",
    ".vision_projection.",
    ".visual_projection.",
)
LANGUAGE_MARKERS = (
    ".model.layers.",
    ".language_model.",
    ".lm_head.",
    ".embed_tokens.",
)


def read_safetensor_keys(adapter_dir: Path) -> list[str]:
    keys: list[str] = []
    files = sorted(adapter_dir.glob("*.safetensors"))
    if not files:
        return keys
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(f"safetensors is required to inspect adapter keys: {exc}")
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())
    return keys


def classify(key: str) -> str:
    low = key.lower()
    if any(marker in low for marker in PROJECTOR_MARKERS):
        return "multimodal_projector_or_merger"
    if any(marker in low for marker in VISION_MARKERS):
        return "vision_tower"
    if any(marker in low for marker in LANGUAGE_MARKERS):
        return "language_model"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument("--require-frozen-vision", action="store_true")
    args = ap.parse_args()

    settings_path = args.settings.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if not adapter_dir.is_dir():
        raise FileNotFoundError(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    keys = read_safetensor_keys(adapter_dir)
    groups: dict[str, list[str]] = {
        "vision_tower": [],
        "multimodal_projector_or_merger": [],
        "language_model": [],
        "other": [],
    }
    for key in keys:
        groups[classify(key)].append(key)

    freeze_cfg = bool(settings.get("training", {}).get("freeze_vision_tower", True))
    issues: list[str] = []
    if not keys:
        issues.append("no_adapter_safetensor_keys_found")
    if args.require_frozen_vision and not freeze_cfg:
        issues.append("settings_freeze_vision_tower_false")
    if args.require_frozen_vision and groups["vision_tower"]:
        issues.append(f"visual_encoder_lora_tensors_detected:{len(groups['vision_tower'])}")
    freeze_projector = bool(settings.get("training", {}).get("freeze_multi_modal_projector", False))
    if freeze_projector and groups["multimodal_projector_or_merger"]:
        issues.append(
            "multimodal_projector_lora_tensors_detected:"
            f"{len(groups['multimodal_projector_or_merger'])}"
        )

    report = {
        "schema_version": "adapter_trainable_scope_v1_2_3",
        "target": args.target,
        "settings": str(settings_path),
        "adapter_dir": str(adapter_dir),
        "adapter_config": str(config_path) if config_path.is_file() else None,
        "freeze_vision_tower_setting": freeze_cfg,
        "freeze_multi_modal_projector_setting": bool(
            settings.get("training", {}).get("freeze_multi_modal_projector", False)
        ),
        "target_modules_from_adapter_config": adapter_config.get("target_modules"),
        "adapter_tensor_count": len(keys),
        "group_counts": {name: len(values) for name, values in groups.items()},
        "group_key_examples": {name: values[:30] for name, values in groups.items()},
        "classification_policy": {
            "vision_markers": list(VISION_MARKERS),
            "projector_markers": list(PROJECTOR_MARKERS),
            "language_markers": list(LANGUAGE_MARKERS),
            "note": "Unknown naming patterns are reported as other and are not silently classified.",
        },
        "issues": issues,
        "passed": not issues,
    }
    out = Path(settings["paths"]["training_run_root"]) / "reports" / f"adapter_trainable_scope_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[ADAPTER SCOPE] {'PASS' if report['passed'] else 'FAIL'} report={out}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
