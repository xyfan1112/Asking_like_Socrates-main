#!/usr/bin/env python3
"""Merge a user's v4 settings into the v4.1 schema without losing local paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import deep_merge, load_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Existing v4 settings.json")
    ap.add_argument("--template", required=True, help="v4.1 settings.example.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    old = load_json(args.old)
    template = load_json(args.template)
    merged = deep_merge(template, old)
    # v4 used a fixed template; v4.1 resolves it per model unless the user has
    # intentionally set a non-default custom value.
    if old.get("training", {}).get("template") == "qwen2_vl_thinking":
        merged["training"]["template"] = "auto"
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"old": str(Path(args.old).resolve()), "template": str(Path(args.template).resolve()), "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
