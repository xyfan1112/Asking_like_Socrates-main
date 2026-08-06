#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def materialize_settings(base_settings: str | Path, profile: str | Path | None = None) -> Path:
    base_path = Path(base_settings).expanduser().resolve()
    settings = load_json(base_path)
    if profile:
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            # Accept either a short profile name (b3_socratic_obb_v1) or an
            # explicit repo-relative path (experiments/b3_socratic_obb_v1.json).
            if profile_path.parts and profile_path.parts[0] == "experiments":
                profile_path = base_path.parent / profile_path
            else:
                profile_path = base_path.parent / "experiments" / profile_path
        if profile_path.suffix == "":
            profile_path = profile_path.with_suffix(".json")
        profile_path = profile_path.expanduser().resolve()
        settings = deep_merge(settings, load_json(profile_path))
        settings.setdefault("experiment", {})["profile_file"] = str(profile_path)
    settings.setdefault("experiment", {})["base_settings"] = str(base_path)
    root = Path(settings["paths"]["results_root"]) / "runtime_settings"
    root.mkdir(parents=True, exist_ok=True)
    name = settings.get("experiment", {}).get("name", "default")
    out = root / f"{name}.json"
    out.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def resolve_python(settings: dict[str, Any], workload: str) -> str:
    runtime = settings.get("runtime", {}).get("workloads", {}).get(workload)
    if runtime is None:
        raise KeyError(f"runtime.workloads does not define workload: {workload}")
    explicit = str(runtime.get("python") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
        raise FileNotFoundError(f"Configured Python does not exist or is not executable: {p}")

    current = Path(sys.executable).resolve()
    env_name = str(runtime.get("conda_env") or "").strip()
    if env_name and (f"/envs/{env_name}/" in str(current) or current.parent.parent.name == env_name):
        return str(current)

    conda = Path(settings.get("runtime", {}).get("conda_executable", "")).expanduser()
    if env_name and conda.is_file():
        env_python = conda.parent.parent / "envs" / env_name / "bin" / "python"
        if env_python.is_file():
            return str(env_python.resolve())

    if bool(runtime.get("allow_fallback_to_current", False)):
        return str(current)
    detail = f"conda_env={env_name!r}" if env_name else "no conda_env configured"
    raise FileNotFoundError(
        f"Cannot resolve Python for workload={workload} ({detail}). "
        "Create the environment, set runtime.workloads.<name>.python, or explicitly enable fallback."
    )


def resolve_model_workload(settings: dict[str, Any], model_key: str, purpose: str) -> str:
    model = settings["models"][model_key]
    configured = model.get("runtime_workloads", {}).get(purpose)
    if configured:
        return str(configured)
    return "sft" if purpose == "sft" else "vllm"


def resolve_model_python(settings: dict[str, Any], model_key: str, purpose: str) -> str:
    return resolve_python(settings, resolve_model_workload(settings, model_key, purpose))


def resolve_llama_factory_dir(settings: dict[str, Any], model_key: str) -> Path:
    model = settings["models"][model_key]
    family = str(model.get("llama_factory_family") or "qwen25")
    directories = settings.get("project", {}).get("llama_factory_dirs", {})
    value = directories.get(family) or settings.get("project", {}).get("llama_factory_dir")
    if not value:
        raise KeyError(f"No LLaMA-Factory directory configured for family={family}")
    return Path(value).expanduser().resolve()


def resolve_training_template(settings: dict[str, Any], model_key: str | None = None) -> str:
    key = model_key or settings["training"]["base_model_key"]
    model = settings["models"][key]
    configured = str(settings["training"].get("template") or "auto")
    if configured != "auto":
        return configured
    template = model.get("training_template")
    if not template:
        raise KeyError(
            f"training.template=auto but models.{key}.training_template is missing"
        )
    return str(template)

def validate_core_settings(settings: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in ("project", "runtime", "paths", "models", "agents", "data_conversion", "trajectory", "training", "evaluation"):
        if key not in settings:
            issues.append(f"missing top-level key: {key}")
    target = settings.get("data_conversion", {}).get("coordinate_target")
    if target not in {"pixel_obb", "norm100_obb", "norm1000_obb", "norm1000_hbb"}:
        issues.append(f"unsupported coordinate_target: {target}")
    return issues
