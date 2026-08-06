#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.settings.read_text(encoding="utf-8"))
    routing = cfg.get("model_routing", {})
    key = str(routing.get("b1_model_key") or "rs_eot_b1_direct")
    entry = cfg.get("models", {}).get(key)
    if not isinstance(entry, dict):
        print(f"[FAIL] settings model entry missing: {key}")
        return 2

    if bool(entry.get("adapter_only") or cfg.get("training", {}).get("adapter_only")):
        adapter = Path(str(entry.get("adapter_path") or ""))
        expected = Path(str(cfg["training"]["b1_direct_standalone_adapter_output"]))
        if adapter != expected:
            print(f"[FAIL] {key}.adapter_path inconsistent: {adapter} != {expected}")
            return 2
        if not (adapter / "adapter_config.json").is_file():
            print("[FAIL] adapter_config.json missing:", adapter / "adapter_config.json")
            return 2
        weights = list(adapter.glob("adapter_model*.safetensors")) + list(adapter.glob("adapter_model*.bin"))
        if not weights:
            print("[FAIL] adapter weights missing:", adapter)
            return 2
        print(f"[PASS] B1 adapter registered: key={key} adapter={adapter}")
        return 0

    expected = Path(cfg["training"]["b1_direct_standalone_merged_output"])
    if Path(str(entry.get("path", ""))) != expected:
        print(f"[FAIL] {key}.path inconsistent")
        return 2
    if not (expected / "config.json").is_file():
        print("[FAIL] B1 merged model incomplete:", expected)
        return 2
    print(f"[PASS] B1 merged model registered: key={key} model={expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
