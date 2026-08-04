#!/usr/bin/env python3
"""Point three logical Socratic roles at one external Qwen3-Omni TP=4 server.

The generated settings must not be passed to the ordinary ``start-agents``
launcher because that launcher starts three independent model processes.  This
file preserves role-specific prompts, max_tokens, temperatures, and all quality
gates while replacing only the physical endpoint and the Full concurrency.
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
    ap.add_argument("--full-concurrency", type=int, default=4)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--checkpoint-id", default="")
    args = ap.parse_args()

    source = args.settings.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.full_concurrency < 1:
        raise ValueError("--full-concurrency must be >=1")
    if args.tensor_parallel_size != 4:
        raise ValueError("v1.2.2 shared Omni topology requires tensor parallel size 4")

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
        role_cfg["deployment_mode"] = "shared_external_omni_tp4_no_tts"
        role_cfg["physical_weight_copy"] = 1
        role_cfg["tensor_parallel_size"] = 4
        # Informational only. Ordinary start-agents must not consume this file.
        role_cfg["gpu"] = 0
        if args.model_path:
            role_cfg["model_path"] = str(Path(args.model_path).expanduser().resolve())

    official = out.setdefault("official_socratic", {})
    original_concurrency = official.get("concurrency")
    official["concurrency"] = args.full_concurrency
    trajectory = out.setdefault("trajectory", {})
    trajectory["concurrency"] = args.full_concurrency

    out["shared_omni_external"] = {
        "enabled": True,
        "base_url": base_url,
        "served_name": args.served_name,
        "model_path": str(Path(args.model_path).expanduser().resolve()) if args.model_path else "",
        "checkpoint_id": args.checkpoint_id,
        "checkpoint_variant": "AWQ-No-TTS",
        "logical_roles": ["reasoner", "perceiver", "verifier"],
        "physical_servers": 1,
        "physical_weight_copies": 1,
        "tensor_parallel_size": 4,
        "all_four_gpus_per_request": True,
        "debug_concurrency": 1,
        "full_concurrency": args.full_concurrency,
        "original_full_concurrency": original_concurrency,
        "forbid_normal_start_agents": True,
        "language_is_selected_by_generation_prompt_not_server": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[SHARED OMNI TP4 SETTINGS] PASS")
    print(" source             =", source)
    print(" output             =", output)
    print(" endpoint           =", base_url)
    print(" served_name        =", args.served_name)
    print(" tensor_parallel    = 4")
    print(" physical_servers   = 1")
    print(" physical_weights   = 1")
    print(" full_concurrency   =", args.full_concurrency)
    print(" debug_concurrency  = 1 (enforced by official debug script)")
    print("[IMPORTANT] Do not run start-agents with this settings file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
