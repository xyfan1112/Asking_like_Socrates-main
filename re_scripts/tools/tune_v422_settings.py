#!/usr/bin/env python3
"""Apply recommended v4.2.2 trajectory settings with an automatic backup."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, default=Path("settings.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = args.settings.resolve()
    if not path.exists():
        raise SystemExit(f"settings not found: {path}")
    settings = json.loads(path.read_text(encoding="utf-8"))

    trajectory = settings.setdefault("trajectory", {})
    official = settings.setdefault("official_socratic", {})
    agents = settings.setdefault("agents", {})
    perceiver = agents.setdefault("perceiver", {})

    changes = {
        "trajectory.prompt_profile": (trajectory.get("prompt_profile"), "obb_grounding_v2"),
        "trajectory.max_reasoning_rounds": (trajectory.get("max_reasoning_rounds"), 8),
        "trajectory.force_final_on_last_round": (trajectory.get("force_final_on_last_round"), True),
        "trajectory.max_repair_attempts": (trajectory.get("max_repair_attempts"), 1),
        "official_socratic.max_loop": (official.get("max_loop"), 8),
        "agents.perceiver.max_tokens": (perceiver.get("max_tokens"), 384),
        "agents.perceiver.temperature": (perceiver.get("temperature"), 0.0),
    }

    trajectory["prompt_profile"] = "obb_grounding_v2"
    trajectory["max_reasoning_rounds"] = 8
    trajectory["force_final_on_last_round"] = True
    trajectory["max_repair_attempts"] = 1
    official["max_loop"] = 8
    perceiver["max_tokens"] = 384
    perceiver["temperature"] = 0.0

    print("[v4.2.2 settings]")
    for key, (old, new) in changes.items():
        marker = "=" if old == new else "->"
        print(f"  {key}: {old!r} {marker} {new!r}")

    model_path = str(perceiver.get("model_path", ""))
    if "rs-eot" in model_path.lower():
        print("[WARN] Perceiver still points to RS-EoT. Prefer Qwen2.5-VL-Instruct for concise atomic perception answers.")

    if args.dry_run:
        print("[DRY RUN] settings not modified")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.backup_v4_2_2_{stamp}")
    shutil.copy2(path, backup)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] updated: {path}")
    print(f"[BACKUP] {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
