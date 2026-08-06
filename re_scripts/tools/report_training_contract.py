#!/usr/bin/env python3
"""Audit the SFT contract before a GPU training process is launched."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def scalar(text: str) -> Any:
    value = text.strip().strip('"\'')
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_flat_yaml(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key and value.strip():
            out[key.strip()] = scalar(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument("--require-frozen-vision", action="store_true")
    args = ap.parse_args()

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    cfg = read_flat_yaml(args.config)
    t = settings["training"]
    world = int(t.get("world_size", 1))
    per_device = int(cfg.get("per_device_train_batch_size", t.get("per_device_train_batch_size", 1)))
    accum = int(cfg.get("gradient_accumulation_steps", t.get("gradient_accumulation_steps", 1)))
    effective = world * per_device * accum
    issues: list[str] = []
    if str(cfg.get("finetuning_type", "lora")) != "lora":
        issues.append("finetuning_type_is_not_lora")
    if args.require_frozen_vision and cfg.get("freeze_vision_tower") is not True:
        issues.append("vision_tower_not_frozen")
    if bool(t.get("freeze_multi_modal_projector", False)) and cfg.get("freeze_multi_modal_projector") is not True:
        issues.append("multimodal_projector_not_frozen")
    active_profile = settings.get("runtime", {}).get("gpu_profiles", {}).get("sft_active", {})
    visible = [x.strip() for x in str(active_profile.get("visible_devices", "")).split(",") if x.strip()]
    expected_world = int(active_profile.get("world_size", world))
    if world != expected_world:
        issues.append(f"world_size_mismatch:settings={world}:profile={expected_world}")
    if visible and len(visible) != world:
        issues.append(f"visible_gpu_count_mismatch:visible={len(visible)}:world={world}")
    if world < 1:
        issues.append(f"invalid_world_size:{world}")
    expected_effective = settings.get("v1_2_3", {}).get("sft", {}).get("effective_global_batch")
    if expected_effective is not None and int(expected_effective) != effective:
        issues.append(f"effective_global_batch_mismatch:actual={effective}:expected={expected_effective}")
    if cfg.get("lora_target") != "all":
        issues.append(f"lora_target_not_all:{cfg.get('lora_target')}")
    if not args.config.is_file():
        issues.append("config_missing")

    report = {
        "schema_version": "training_contract_v1_2_3",
        "target": args.target,
        "settings": str(args.settings),
        "config": str(args.config),
        "passed": not issues,
        "issues": issues,
        "training": {
            "finetuning_type": cfg.get("finetuning_type"),
            "lora_target": cfg.get("lora_target"),
            "lora_rank": cfg.get("lora_rank"),
            "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "freeze_vision_tower": cfg.get("freeze_vision_tower"),
            "freeze_multi_modal_projector": cfg.get("freeze_multi_modal_projector"),
            "world_size": world,
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": accum,
            "effective_global_batch": effective,
            "base_model": cfg.get("model_name_or_path"),
            "output_dir": cfg.get("output_dir"),
            "dataset": cfg.get("dataset"),
        },
        "interpretation": {
            "vision_tower": "frozen" if cfg.get("freeze_vision_tower") is True else "trainable_or_unconfirmed",
            "multimodal_projector": "not_frozen_by_flag" if cfg.get("freeze_multi_modal_projector") is False else "frozen",
            "actual_trainable_parameter_counts": "Extracted from the LLaMA-Factory runtime log after model construction.",
        },
    }
    root = Path(settings["paths"]["training_run_root"])
    out = root / "reports" / f"training_contract_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    txt = out.with_suffix(".txt")
    txt.write_text(
        "\n".join(
            [
                f"target={args.target}",
                f"passed={report['passed']}",
                f"finetuning_type={cfg.get('finetuning_type')}",
                f"lora_target={cfg.get('lora_target')}",
                f"freeze_vision_tower={cfg.get('freeze_vision_tower')}",
                f"freeze_multi_modal_projector={cfg.get('freeze_multi_modal_projector')}",
                f"world_size={world}",
                f"gradient_accumulation_steps={accum}",
                f"effective_global_batch={effective}",
                f"issues={issues}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[TRAIN CONTRACT] report={out}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
