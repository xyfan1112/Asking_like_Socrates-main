#!/usr/bin/env python3
"""Migrate an existing v4.1/v4.2 settings file onto the v4.3 template."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "settings.example.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    old_path = Path(args.old)
    template_path = Path(args.template)
    output_path = Path(args.output)
    old = json.loads(old_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    merged = deep_merge(template, old)

    trajectory = merged.setdefault("trajectory", {})
    trajectory.update({
        "prompt_profile": "obb_grounding_v3",
        "force_final_on_last_round": True,
        "max_repair_attempts": max(2, int(trajectory.get("max_repair_attempts", 0))),
        "min_strict_grounding_samples": max(20, int(trajectory.get("min_strict_grounding_samples", 0))),
        "min_strict_classification_samples": max(20, int(trajectory.get("min_strict_classification_samples", 0))),
    })
    trajectory.setdefault("min_perception_rounds_by_task", {
        "ref_grounding_obb": 2,
        "ref_classification": 1,
    })
    merged.setdefault("official_socratic", {})["max_loop"] = max(
        8, int(merged["official_socratic"].get("max_loop", 0))
    )
    merged.setdefault("training", {})["require_agents_stopped"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output_path.exists():
        backup = output_path.with_name(
            output_path.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.copyfile(output_path, backup)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SETTINGS MIGRATION] PASS")
    print(f"  old: {old_path}")
    print(f"  template: {template_path}")
    print(f"  output: {output_path}")
    print(f"  backup: {backup}")
    print("  enforced: prompt_profile=obb_grounding_v3 max_loop>=8 repair_attempts>=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
