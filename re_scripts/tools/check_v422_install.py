#!/usr/bin/env python3
"""Check that the active re_scripts tree is a coherent v4.2.2 installation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def contains(path: Path, text: str) -> bool:
    return path.is_file() and text in path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    checks = {
        "strict_data_gate": (root / "data_layer/05b_verify_data_outputs.py").is_file(),
        "nul_launcher": contains(root / "data_layer/official_socratic/02_run_official_generation.sh", "mapfile -d '' -t CFG"),
        "structured_rewrite_bypass": contains(root / "data_layer/official_socratic/local_api_adapter/utils.py", "structured_rewrite_bypass"),
        "token_logging": contains(root / "data_layer/official_socratic/local_api_adapter/utils.py", "finish_reason"),
        "obb_grounding_v2": contains(root / "data_layer/prompts/profiles.py", '"obb_grounding_v2"'),
        "debug_audit": (root / "data_layer/official_socratic/02b_audit_debug_generation.py").is_file(),
    }
    version = None
    version_path = root / "VERSION.json"
    if version_path.exists():
        version = json.loads(version_path.read_text(encoding="utf-8")).get("version")
    checks["version_4_2_2"] = version == "4.2.2"

    settings_path = args.settings or (root / "settings.json")
    setting_summary = None
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        setting_summary = {
            "prompt_profile": settings.get("trajectory", {}).get("prompt_profile"),
            "trajectory_max_rounds": settings.get("trajectory", {}).get("max_reasoning_rounds"),
            "official_max_loop": settings.get("official_socratic", {}).get("max_loop"),
            "force_final": settings.get("trajectory", {}).get("force_final_on_last_round"),
            "repair_attempts": settings.get("trajectory", {}).get("max_repair_attempts"),
            "perceiver_max_tokens": settings.get("agents", {}).get("perceiver", {}).get("max_tokens"),
        }

    passed = all(checks.values())
    print(f"[v4.2.2 INSTALL CHECK] {'PASS' if passed else 'FAIL'}")
    for name, value in checks.items():
        print(f"  {name}: {'OK' if value else 'MISSING'}")
    if setting_summary is not None:
        print("  settings:")
        for key, value in setting_summary.items():
            print(f"    {key}: {value!r}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
