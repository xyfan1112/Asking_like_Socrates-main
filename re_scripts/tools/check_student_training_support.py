#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from config import resolve_llama_factory_dir, resolve_model_python, resolve_training_template  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    args = ap.parse_args()
    s = json.loads(args.settings.read_text(encoding="utf-8"))
    route = s.get("model_routing", {})
    key = str(route.get("b0_model_key") or s["training"]["base_model_key"])
    model = s["models"][key]
    issues: list[str] = []

    model_path = Path(str(model["path"]))
    if not (model_path / "config.json").is_file():
        issues.append(f"model_config_missing:{model_path / 'config.json'}")
    template = resolve_training_template(s, key)
    if route.get("student_family") == "qwen3_omni" and template != "qwen3_omni":
        issues.append(f"wrong_template:{template}")
    if route.get("student_family") == "qwen3_omni":
        if not bool(s["training"].get("adapter_only")):
            issues.append("qwen3_omni_requires_adapter_only")
        if not bool(s["training"].get("freeze_vision_tower")):
            issues.append("vision_tower_not_frozen")
        if not bool(s["training"].get("freeze_multi_modal_projector")):
            issues.append("multimodal_projector_not_frozen")

    lf_root = resolve_llama_factory_dir(s, key)
    lf_src = lf_root / "src" / "llamafactory"
    if not lf_src.is_dir():
        issues.append(f"llamafactory_source_missing:{lf_src}")
    elif route.get("student_family") == "qwen3_omni":
        found = False
        for path in lf_src.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "qwen3_omni" in text or "Qwen3Omni" in text:
                found = True
                break
        if not found:
            issues.append("local_llamafactory_has_no_qwen3_omni_support")

    sft_python = Path(resolve_model_python(s, key, "sft"))
    if not sft_python.is_file():
        issues.append(f"sft_python_missing:{sft_python}")
    elif route.get("student_family") == "qwen3_omni" and (model_path / "config.json").is_file():
        probe = """
import sys
from transformers import AutoConfig, Qwen3OmniMoeProcessor
root=sys.argv[1]
cfg=AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
processor=Qwen3OmniMoeProcessor.from_pretrained(root, trust_remote_code=True, local_files_only=True)
print(type(cfg).__name__, type(processor).__name__)
"""
        proc = subprocess.run(
            [str(sft_python), "-c", probe, str(model_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            issues.append("sft_environment_cannot_load_qwen3_omni_config_or_processor:" + proc.stdout[-1200:].replace("\n", " | "))

    report = {
        "schema_version": "student_training_preflight_v1",
        "student_family": route.get("student_family"),
        "model_key": key,
        "model_path": str(model_path),
        "template": template,
        "llama_factory_dir": str(lf_root),
        "sft_python": str(sft_python),
        "adapter_only": bool(s["training"].get("adapter_only")),
        "freeze_vision_tower": bool(s["training"].get("freeze_vision_tower")),
        "freeze_multi_modal_projector": bool(s["training"].get("freeze_multi_modal_projector")),
        "issues": issues,
        "passed": not issues,
    }
    out = Path(s["paths"]["training_run_root"]) / "reports" / "student_training_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[STUDENT TRAIN PREFLIGHT]", "PASS" if not issues else "FAIL", "report=", out)
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
