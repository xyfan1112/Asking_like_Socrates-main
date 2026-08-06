#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_json, resolve_python


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--field", choices=["python", "bin_dir", "conda_env"], default="python")
    args = ap.parse_args()
    settings = load_json(args.settings)
    if args.workload not in settings.get("runtime", {}).get("workloads", {}):
        available = ", ".join(sorted(settings.get("runtime", {}).get("workloads", {})))
        raise SystemExit(f"Unknown workload {args.workload!r}. Available: {available}")
    python_bin = Path(resolve_python(settings, args.workload))
    if args.field == "python":
        print(python_bin)
    elif args.field == "bin_dir":
        print(python_bin.parent)
    else:
        print(
            settings.get("runtime", {})
            .get("workloads", {})
            .get(args.workload, {})
            .get("conda_env", "")
        )


if __name__ == "__main__":
    main()
