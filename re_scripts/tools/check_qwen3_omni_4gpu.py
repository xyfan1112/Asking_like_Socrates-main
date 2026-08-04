#!/usr/bin/env python3
"""Offline preflight for Qwen3-Omni on four A6000 GPUs.

This tool deliberately does not install packages or launch a model.  It checks
hardware visibility, package presence, model metadata and the GPU topology so a
separate Omni environment can be prepared without touching the stable v1.2.2.2
agent environment.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--model-path", type=Path)
    ap.add_argument(
        "--server-mode",
        choices=("vllm_thinker", "vllm_omni"),
        default="vllm_thinker",
        help="vllm_thinker uses ordinary vLLM TP=4 and text output only; vllm_omni uses the multi-stage server",
    )
    args = ap.parse_args()
    critical: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {}

    try:
        import torch
        count = torch.cuda.device_count()
        gpus = []
        for index in range(count):
            prop = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": prop.name,
                    "memory_gib": round(prop.total_memory / 1024**3, 2),
                    "compute_capability": f"{prop.major}.{prop.minor}",
                }
            )
        report["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "gpu_count": count,
            "gpus": gpus,
        }
        if count < 4:
            critical.append(f"visible_gpu_count={count}<4")
        for gpu in gpus[:4]:
            if float(gpu["memory_gib"]) < 44:
                critical.append(f"gpu{gpu['index']}_memory={gpu['memory_gib']}GiB<44GiB")
            if "A6000" not in str(gpu["name"]):
                warnings.append(f"gpu{gpu['index']}_is_not_A6000:{gpu['name']}")
    except Exception as exc:
        critical.append(f"torch_gpu_check_failed:{type(exc).__name__}:{exc}")

    packages = {
        name: version(name)
        for name in (
            "vllm",
            "vllm-omni",
            "transformers",
            "compressed-tensors",
            "llamafactory",
            "peft",
            "accelerate",
        )
    }
    report["packages"] = packages
    report["server_mode"] = args.server_mode
    if not packages.get("vllm"):
        critical.append("vllm_not_installed_in_this_environment")
    if args.server_mode == "vllm_omni" and not packages.get("vllm-omni"):
        critical.append("vllm-omni_not_installed_for_vllm_omni_mode")
    if not packages.get("transformers"):
        critical.append("transformers_not_installed")

    model_info: dict[str, object] = {}
    if args.model_path:
        model = args.model_path.expanduser().resolve()
        model_info["path"] = str(model)
        config_path = model / "config.json"
        if not config_path.is_file():
            critical.append(f"model_config_missing:{config_path}")
        else:
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                model_info["architectures"] = cfg.get("architectures")
                model_info["model_type"] = cfg.get("model_type")
                quant = cfg.get("quantization_config")
                model_info["quantization_config"] = quant
                architectures = " ".join(cfg.get("architectures") or [])
                if "Qwen3Omni" not in architectures and "qwen3_omni" not in str(cfg.get("model_type", "")):
                    critical.append("model_is_not_recognized_as_Qwen3_Omni")
                if quant:
                    method = str(quant.get("quant_method") or quant.get("format") or "unknown")
                    warnings.append(
                        "prequantized_checkpoint_detected:" + method
                        + "; exact vLLM-Omni kernel compatibility must be proven by a one-request smoke test"
                    )
            except Exception as exc:
                critical.append(f"model_config_invalid:{type(exc).__name__}:{exc}")
        weight_bytes = sum(
            path.stat().st_size
            for path in model.rglob("*")
            if path.is_file() and path.suffix in {".safetensors", ".bin"}
        )
        model_info["local_weight_gib"] = round(weight_bytes / 1024**3, 2)
    else:
        warnings.append("model_path_not_provided")
    report["model"] = model_info

    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        report["topology"] = topo.stdout.strip() or topo.stderr.strip()
        if topo.returncode != 0:
            warnings.append("nvidia_smi_topology_failed")
    except Exception as exc:
        warnings.append(f"nvidia_smi_topology_exception:{type(exc).__name__}:{exc}")

    report["recommended_architecture"] = {
        "trajectory_generation": (
            "one shared Qwen3-Omni thinker/text server distributed over four GPUs; "
            "Reasoner/Perceiver/Verifier remain three prompt roles, not three weight copies. "
            "For image+text input and text-only trajectories, ordinary vLLM thinker serving "
            "with tensor parallel size 4 is the conservative default."
        ),
        "training": (
            "LoRA/QLoRA in a separate current environment; save adapter first. "
            "For a standalone merged quantized checkpoint, merge into an unquantized base "
            "and quantize the merged model again."
        ),
    }
    report["critical"] = critical
    report["warnings"] = warnings
    report["passed"] = not critical
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OMNI PREFLIGHT] {'PASS' if not critical else 'FAIL'}")
    if warnings:
        print(f"[OMNI PREFLIGHT] warnings={len(warnings)}")
    return 0 if not critical else 2


if __name__ == "__main__":
    raise SystemExit(main())
