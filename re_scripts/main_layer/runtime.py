#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    load_json,
    resolve_model_workload,
    resolve_python,
    validate_core_settings,
)


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) != 0


def _module_for_workload(name: str, cfg: dict[str, Any]) -> str:
    explicit = str(cfg.get("required_module") or "").strip()
    if explicit:
        return explicit
    if "vllm" in name:
        return "vllm"
    if "sft" in name:
        return "llamafactory"
    if name == "rl":
        return "verl"
    return "datasets"


def probe_python(python_bin: str, module: str) -> dict[str, Any]:
    code = r'''
import importlib.util,json,sys
m=sys.argv[1]
result={"python":sys.executable,"module":m,"found":importlib.util.find_spec(m) is not None}
for pkg in ("torch","transformers","vllm","llamafactory","peft"):
    try:
        mod=__import__(pkg)
        result[pkg+"_version"]=getattr(mod,"__version__","unknown")
    except Exception as exc:
        result[pkg+"_error"]=f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
'''
    out = subprocess.check_output([python_bin, "-c", code, module], text=True)
    return json.loads(out)


def _version_at_least(value: str | None, minimum: str | None) -> bool | None:
    if not value or not minimum:
        return None
    try:
        from packaging.version import Version

        return Version(value.split("+")[0]) >= Version(minimum)
    except Exception:
        return None


def model_compatibility(
    name: str,
    cfg: dict[str, Any],
    workloads: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    architecture = str(cfg.get("architecture") or "")
    checks: dict[str, Any] = {
        "architecture": architecture,
        "runtime_workloads": cfg.get("runtime_workloads", {}),
        "requirements": cfg.get("minimum_versions", {}),
        "issues": [],
    }
    for purpose in ("vllm", "sft"):
        workload = resolve_model_workload(settings, name, purpose)
        probe = workloads.get(workload, {})
        checks[f"{purpose}_workload"] = workload
        if "error" in probe or not probe.get("found", False):
            checks["issues"].append(f"{purpose} workload unavailable: {workload}")
            continue
        minimums = cfg.get("minimum_versions", {})
        if purpose == "vllm":
            for pkg in ("vllm", "transformers"):
                minimum = minimums.get(pkg)
                if minimum:
                    actual = probe.get(pkg + "_version")
                    ok = _version_at_least(actual, minimum)
                    checks[f"{pkg}_ok_for_{purpose}"] = ok
                    if ok is False:
                        checks["issues"].append(
                            f"{workload}: {pkg}={actual}, requires >= {minimum}"
                        )
        else:
            minimum = minimums.get("transformers")
            if minimum:
                actual = probe.get("transformers_version")
                ok = _version_at_least(actual, minimum)
                checks["transformers_ok_for_sft"] = ok
                if ok is False:
                    checks["issues"].append(
                        f"{workload}: transformers={actual}, requires >= {minimum}"
                    )
    checks["compatible"] = not checks["issues"]
    if architecture == "qwen3_vl" and checks["issues"]:
        checks["recommendation"] = (
            "Keep Qwen2.5 in als_sft/als_vllm. Create separate als_qwen3_sft and "
            "als_qwen3_vllm environments; Qwen3-VL officially needs transformers>=4.57.0 "
            "and vLLM>=0.11.0. Do not upgrade the working Qwen2.5 environments in place."
        )
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Check environments, GPUs, paths, ports and model-family compatibility.")
    ap.add_argument("--settings", required=True)
    args = ap.parse_args()
    settings = load_json(args.settings)
    report: dict[str, Any] = {
        "settings_issues": validate_core_settings(settings),
        "workloads": {},
        "models": {},
        "model_compatibility": {},
        "ports": {},
        "gpu": {},
    }

    for workload, cfg in settings.get("runtime", {}).get("workloads", {}).items():
        module = _module_for_workload(workload, cfg)
        try:
            python_bin = resolve_python(settings, workload)
            report["workloads"][workload] = probe_python(python_bin, module)
        except Exception as exc:
            report["workloads"][workload] = {"error": f"{type(exc).__name__}: {exc}"}

    for name, cfg in settings.get("models", {}).items():
        path = Path(str(cfg.get("path", ""))).expanduser()
        report["models"][name] = {
            "path": str(path),
            "directory": path.is_dir(),
            "config": (path / "config.json").is_file(),
            "architecture": cfg.get("architecture"),
        }
        report["model_compatibility"][name] = model_compatibility(
            name, cfg, report["workloads"], settings
        )

    for role, cfg in settings.get("agents", {}).items():
        host, port = cfg.get("host", "127.0.0.1"), int(cfg.get("port", 0))
        report["ports"][role] = {
            "host": host,
            "port": port,
            "free": port_free(host, port),
            "meaning": "free=false is normal when the agent service is already running",
        }

    try:
        import torch

        report["gpu"] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count(),
            "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        report["gpu"] = {"error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    bad = bool(report["settings_issues"]) or not report.get("gpu", {}).get("available", False)
    raise SystemExit(2 if bad else 0)


if __name__ == "__main__":
    main()
