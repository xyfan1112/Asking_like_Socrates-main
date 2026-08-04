#!/usr/bin/env python3
"""Point three logical Socratic roles at one externally managed Omni server.

This tool does not start a model and does not alter the input settings.  The
normal ``start-agents`` command must not be used with the generated settings,
because the 2A6000 launcher starts three separate ordinary vLLM processes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from urllib.parse import urlparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--text-only", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    source = args.settings.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    cfg = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("settings root must be a JSON object")

    parsed = urlparse(args.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid --base-url: {args.base_url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError("--base-url must end with /v1")

    out = copy.deepcopy(cfg)
    roles = out.get("agents")
    if not isinstance(roles, dict):
        raise KeyError("settings.agents is required")
    for role in ("reasoner", "perceiver", "verifier"):
        role_cfg = roles.get(role)
        if not isinstance(role_cfg, dict):
            raise KeyError(f"settings.agents.{role} is required")
        role_cfg["base_url"] = base_url
        role_cfg["host"] = parsed.hostname
        role_cfg["port"] = port
        role_cfg["served_name"] = args.served_name
        role_cfg["omni_text_only"] = bool(args.text_only)
        role_cfg["deployment_mode"] = "shared_external_omni"
        # Preserve max_tokens/temperature and every scientific parameter.
        # The field is informational because this settings file must not be
        # passed to the ordinary three-process start-agents launcher.
        role_cfg["gpu"] = 0
        if args.model_path:
            role_cfg["model_path"] = str(Path(args.model_path).expanduser().resolve())

    out["shared_omni_external"] = {
        "enabled": True,
        "base_url": base_url,
        "served_name": args.served_name,
        "model_path": str(Path(args.model_path).expanduser().resolve()) if args.model_path else "",
        "text_only": bool(args.text_only),
        "logical_roles": ["reasoner", "perceiver", "verifier"],
        "physical_weight_copies": 1,
        "forbid_normal_start_agents": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[SHARED OMNI SETTINGS] PASS")
    print(" source      =", source)
    print(" output      =", output)
    print(" endpoint    =", base_url)
    print(" served_name =", args.served_name)
    print(" text_only   =", bool(args.text_only))
    print("[IMPORTANT] Do not run start-agents with this settings file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
