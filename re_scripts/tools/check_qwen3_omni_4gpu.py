#!/usr/bin/env python3
"""Fail-closed preflight for Qwen3-Omni AWQ-No-TTS on one 4-GPU TP server.

The tool does not download, install, start, or modify anything. It checks:
- exactly usable four-GPU visibility and memory;
- vLLM CLI support for the required tensor-parallel flags;
- Qwen3-Omni model metadata and pre-quantization fields;
- whether the checkpoint index contains Talker/TTS weights;
- approximate per-GPU weight share and inter-GPU topology.

A PASS means that the static contract is plausible. It is not a substitute for
an actual text+image smoke test because third-party AWQ kernel compatibility is
resolved only while loading and executing the checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be object: {path}")
    return value


def find_weight_index(model: Path) -> Path | None:
    candidates = (
        model / "model.safetensors.index.json",
        model / "model-00001-of-00001.safetensors.index.json",
        model / "pytorch_model.bin.index.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    found = sorted(model.glob("*.index.json"))
    return found[0] if found else None


def inspect_weight_keys(model: Path) -> dict[str, Any]:
    index_path = find_weight_index(model)
    result: dict[str, Any] = {
        "index_path": str(index_path) if index_path else None,
        "key_count": None,
        "prefix_counts": {},
        "talker_keys": [],
        "thinker_keys": [],
        "visual_keys": [],
        "audio_keys": [],
    }
    if index_path is None:
        return result
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return result
    keys = [str(key) for key in weight_map]
    result["key_count"] = len(keys)
    patterns = {
        "talker_keys": re.compile(r"(?:^|\.)(?:talker|code2wav|tts|speaker_encoder)(?:\.|$)", re.I),
        "thinker_keys": re.compile(r"(?:^|\.)(?:thinker)(?:\.|$)", re.I),
        "visual_keys": re.compile(r"(?:visual|vision|image)", re.I),
        "audio_keys": re.compile(r"(?:audio|speech)", re.I),
    }
    for name, pattern in patterns.items():
        matches = [key for key in keys if pattern.search(key)]
        result[name] = matches[:20]
        result[name.replace("_keys", "_count")] = len(matches)
    prefix_counts: dict[str, int] = {}
    for key in keys:
        prefix = ".".join(key.split(".")[:2])
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    result["prefix_counts"] = dict(
        sorted(prefix_counts.items(), key=lambda item: (-item[1], item[0]))[:30]
    )
    return result


def run_text(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def inspect_vllm_cli(python_exe: str) -> dict[str, Any]:
    vllm_bin = Path(python_exe).resolve().parent / "vllm"
    result: dict[str, Any] = {
        "python": python_exe,
        "vllm_bin": str(vllm_bin),
        "vllm_bin_exists": vllm_bin.is_file(),
        "flags": {},
    }
    if vllm_bin.is_file():
        code, text = run_text([str(vllm_bin), "serve", "--help"], timeout=60)
    else:
        code, text = run_text([python_exe, "-m", "vllm.entrypoints.cli.main", "serve", "--help"], timeout=60)
    result["help_returncode"] = code
    result["help_preview"] = text[:4000]
    for flag in (
        "--tensor-parallel-size",
        "--distributed-executor-backend",
        "--max-model-len",
        "--max-num-seqs",
        "--gpu-memory-utilization",
        "--limit-mm-per-prompt",
        "--allowed-local-media-path",
        "--quantization",
        "--trust-remote-code",
        "--enforce-eager",
        "--enable-prefix-caching",
    ):
        result["flags"][flag] = flag in text
    return result


def gpu_report() -> tuple[dict[str, Any], list[str], list[str]]:
    critical: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {}
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
        if not torch.cuda.is_available():
            critical.append("cuda_not_available")
        if count < 4:
            critical.append(f"visible_gpu_count={count}<4")
        for gpu in gpus[:4]:
            if float(gpu["memory_gib"]) < 44:
                critical.append(f"gpu{gpu['index']}_memory={gpu['memory_gib']}GiB<44GiB")
            if "A6000" not in str(gpu["name"]):
                warnings.append(f"gpu{gpu['index']}_is_not_A6000:{gpu['name']}")
    except Exception as exc:
        critical.append(f"torch_gpu_check_failed:{type(exc).__name__}:{exc}")

    code, topo = run_text(["nvidia-smi", "topo", "-m"], timeout=20)
    report["topology"] = topo.strip()
    if code != 0:
        warnings.append("nvidia_smi_topology_failed")
    else:
        # TP=4 still works over PCIe, but SYS/PHB links can make TP latency
        # slower than a smaller TP topology. This is advisory, not a failure.
        if "NV" not in topo and any(token in topo for token in ("SYS", "PHB", "PXB")):
            warnings.append("four_gpu_topology_has_no_visible_nvlink;TP4_uses_PCIe_and_may_be_communication_bound")
    return report, critical, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--model-id", default="tclf90/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS")
    ap.add_argument("--python", dest="python_exe", default=sys.executable)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--require-no-tts", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    critical: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "model_id": args.model_id,
        "tensor_parallel_size": args.tensor_parallel_size,
        "require_no_tts": args.require_no_tts,
    }

    gpu, gpu_critical, gpu_warnings = gpu_report()
    report.update(gpu)
    critical.extend(gpu_critical)
    warnings.extend(gpu_warnings)
    if args.tensor_parallel_size != 4:
        critical.append(f"tensor_parallel_size={args.tensor_parallel_size};v1.2.2_requires_4")

    packages = {
        name: version(name)
        for name in (
            "vllm",
            "vllm-omni",
            "transformers",
            "compressed-tensors",
            "torch",
        )
    }
    report["packages"] = packages
    if not packages.get("vllm"):
        critical.append("vllm_not_installed_in_omni_environment")
    if not packages.get("transformers"):
        critical.append("transformers_not_installed_in_omni_environment")

    cli = inspect_vllm_cli(args.python_exe)
    report["vllm_cli"] = cli
    if cli.get("help_returncode") != 0:
        critical.append("vllm_serve_help_failed")
    for required in (
        "--tensor-parallel-size",
        "--max-model-len",
        "--max-num-seqs",
        "--gpu-memory-utilization",
    ):
        if not cli.get("flags", {}).get(required):
            critical.append(f"vllm_cli_missing_required_flag:{required}")
    for useful in ("--limit-mm-per-prompt", "--allowed-local-media-path"):
        if not cli.get("flags", {}).get(useful):
            warnings.append(f"vllm_cli_missing_optional_multimodal_flag:{useful}")

    model = args.model_path.expanduser().resolve()
    model_info: dict[str, Any] = {"path": str(model), "exists": model.is_dir()}
    if not model.is_dir():
        critical.append(f"model_path_missing:{model}")
    else:
        config_path = model / "config.json"
        model_info["config_path"] = str(config_path)
        if not config_path.is_file():
            critical.append(f"model_config_missing:{config_path}")
        else:
            try:
                cfg = read_json(config_path)
                model_info["architectures"] = cfg.get("architectures")
                model_info["model_type"] = cfg.get("model_type")
                model_info["torch_dtype"] = cfg.get("torch_dtype")
                quant = cfg.get("quantization_config")
                model_info["quantization_config"] = quant
                architecture_text = " ".join(str(x) for x in (cfg.get("architectures") or []))
                model_type = str(cfg.get("model_type") or "")
                if "Qwen3Omni" not in architecture_text and "qwen3_omni" not in model_type.lower():
                    critical.append("model_is_not_recognized_as_Qwen3_Omni")
                if not isinstance(quant, dict):
                    warnings.append("quantization_config_missing;checkpoint_name_says_AWQ_but_config_does_not_prove_it")
                else:
                    method = str(quant.get("quant_method") or quant.get("format") or "unknown").lower()
                    bits = quant.get("bits")
                    group_size = quant.get("group_size")
                    model_info["quant_summary"] = {
                        "method": method,
                        "bits": bits,
                        "group_size": group_size,
                    }
                    if "awq" not in method:
                        warnings.append(f"quantization_method_is_not_explicit_awq:{method}")
                    warnings.append(
                        "prequantized_checkpoint_requires_runtime_load_and_image_smoke;"
                        "static_metadata_cannot_prove_kernel_compatibility"
                    )
            except Exception as exc:
                critical.append(f"model_config_invalid:{type(exc).__name__}:{exc}")

        weight_files = [
            path
            for path in model.rglob("*")
            if path.is_file() and path.suffix in {".safetensors", ".bin"}
        ]
        weight_bytes = sum(path.stat().st_size for path in weight_files)
        weight_gib = weight_bytes / 1024**3
        model_info["weight_file_count"] = len(weight_files)
        model_info["local_weight_gib"] = round(weight_gib, 2)
        model_info["approx_weight_share_per_gpu_gib"] = round(weight_gib / max(args.tensor_parallel_size, 1), 2)
        if not weight_files:
            critical.append("no_local_weight_files_found")
        if weight_gib / max(args.tensor_parallel_size, 1) > 38:
            warnings.append("weight_share_per_gpu_is_large;KV_and_multimodal_modules_may_leave_insufficient_headroom")

        key_info = inspect_weight_keys(model)
        model_info["weight_index_inspection"] = key_info
        talker_count = int(key_info.get("talker_count") or 0)
        thinker_count = int(key_info.get("thinker_count") or 0)
        no_index = key_info.get("index_path") is None
        if talker_count > 0:
            critical.append(f"talker_or_tts_weights_detected:{talker_count}")
        elif args.require_no_tts and no_index:
            critical.append(
                "cannot_prove_no_tts_without_weight_index;set_OMNI_REQUIRE_NO_TTS=0_only_after_manual_confirmation"
            )
        elif args.require_no_tts and thinker_count == 0:
            warnings.append("weight_index_has_no_explicit_thinker_prefix;verify_checkpoint_layout_manually")
        else:
            model_info["no_tts_static_check"] = "passed_no_talker_keys_detected"

    report["model"] = model_info
    report["recommended_topology"] = {
        "physical_servers": 1,
        "physical_weight_copies": 1,
        "logical_roles": ["reasoner", "perceiver", "verifier"],
        "tensor_parallel_size": 4,
        "each_request_uses_all_four_gpus": True,
        "debug_concurrency": 1,
        "full_concurrency_initial": 4,
        "note": (
            "TP=4 shards one model across all four GPUs. It maximizes usable memory and lets every "
            "request execute across four cards. It does not guarantee fourfold latency speedup because "
            "all-reduce communication can dominate on PCIe-only topology."
        ),
    }
    report["critical"] = critical
    report["warnings"] = warnings
    report["passed"] = not critical
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OMNI TP4 PREFLIGHT] {'PASS' if not critical else 'FAIL'}")
    if warnings:
        print(f"[OMNI TP4 PREFLIGHT] warnings={len(warnings)}")
    return 0 if not critical else 2


if __name__ == "__main__":
    raise SystemExit(main())
