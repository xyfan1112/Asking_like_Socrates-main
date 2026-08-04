#!/usr/bin/env python3
"""Create an isolated Chinese settings file while leaving English untouched."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _abs(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-settings", required=True, type=Path)
    ap.add_argument("--classes-file", required=True, type=Path)
    ap.add_argument("--lang", choices=("zh", "en"), required=True)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    base_path = args.base_settings.expanduser().resolve()
    classes_path = args.classes_file.expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not classes_path.is_file():
        raise FileNotFoundError(classes_path)

    if args.lang == "en":
        print("[ENGLISH UNCHANGED] Use the original settings file directly:")
        print(base_path)
        return 0

    if args.output_root is None:
        raise SystemExit("--output-root is required for --lang zh")
    output_root = args.output_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else output_root / "config" / "settings.zh.v1.2.0.json"
    )

    base = json.loads(base_path.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)

    # Scientific controls are invariants in this patch, not tuning knobs.
    threshold = float(cfg.get("trajectory", {}).get("question_similarity_threshold", 0.95))
    if abs(threshold - 0.95) > 1e-12:
        raise SystemExit(
            "Base settings question_similarity_threshold is not 0.95. "
            "This package refuses to silently change it; fix/select the intended stable base settings first."
        )

    paths = cfg.setdefault("paths", {})
    input_dota_root = paths.get("dota128_root")
    if not input_dota_root:
        raise KeyError("paths.dota128_root is required")

    # Keep original input data and model/runtime/project paths. Redirect only
    # generated Chinese artifacts under one user-selected root.
    generated = {
        "datasets_root": output_root / "datasets",
        "results_root": output_root / "results",
        "dota128_ref_root": output_root / "datasets" / "ref",
        "dota128_llamafactory_root": output_root / "datasets" / "llamafactory",
        "dota128_rl_root": output_root / "datasets" / "rl",
        "pipeline_work_root": output_root / "pipeline",
        "trajectory_raw_dir": output_root / "pipeline" / "trajectory_raw",
        "trajectory_audit_dir": output_root / "pipeline" / "trajectory_audit",
        "training_run_root": output_root / "training",
        "rl_run_root": output_root / "rl",
        "test_run_root": output_root / "testing",
    }
    for key, value in generated.items():
        paths[key] = str(value)
    paths["dota128_root"] = str(input_dota_root)

    official = cfg.setdefault("official_socratic", {})
    official["input_parquet_dir"] = str(output_root / "pipeline" / "official_socratic" / "input")
    official["raw_output_dir"] = str(output_root / "pipeline" / "official_socratic" / "raw")
    official["postproc_output_dir"] = str(output_root / "pipeline" / "official_socratic" / "postproc")

    cfg["taxonomy"] = {
        "mode": "custom",
        "dataset_name": "custom_obb_zh_v1.2.0",
        "classes_file": str(classes_path),
        "qa_language": "zh",
        "strict_exact_final_label": True,
        "preserve_output_paths": True,
    }
    cfg.setdefault("data_conversion", {})["qa_language"] = "zh"
    trajectory = cfg.setdefault("trajectory", {})
    trajectory["qa_language"] = "zh"
    trajectory["prompt_profile"] = "custom_obb_bilingual_v1"
    # Deliberately preserve threshold, rounds and every model token budget.
    trajectory["question_similarity_threshold"] = threshold

    training = cfg.setdefault("training", {})
    training["b1_adapter_output"] = str(output_root / "training" / "b1_direct_lora")
    training["b2_adapter_output"] = str(output_root / "training" / "b2_socratic_lora")
    training["b1_merged_output"] = str(output_root / "models" / "RS-EoT-Custom-B1-Direct-Merged")
    training["b2_merged_output"] = str(output_root / "models" / "RS-EoT-Custom-B2-Socratic-Merged")
    training["d1_adapter_output"] = str(output_root / "training" / "d1_direct_only_lora")
    training["d1_merged_output"] = str(output_root / "models" / "RS-EoT-Custom-D1-Direct-Only-Merged")

    base_key = training.get("base_model_key", "rs_eot")
    base_model = copy.deepcopy(cfg.get("models", {}).get(base_key, {}))
    if not base_model:
        raise KeyError(f"models.{base_key} is required")
    base_model["path"] = training["d1_merged_output"]
    base_model["served_name"] = "rs-eot-d1-direct-only"
    cfg.setdefault("models", {})["rs_eot_d1_direct_only"] = base_model

    cfg["v1_2_0_invariants"] = {
        "english_settings_unchanged": str(base_path),
        "chinese_output_root": str(output_root),
        "input_dota128_root_unchanged": str(input_dota_root),
        "question_similarity_threshold": threshold,
        "official_max_loop_preserved": official.get("max_loop"),
        "agent_max_tokens_preserved": {
            role: cfg.get("agents", {}).get(role, {}).get("max_tokens")
            for role in ("reasoner", "perceiver", "verifier")
        },
    }

    for path in [output_root, output.parent, *generated.values()]:
        Path(path).mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(cfg["v1_2_0_invariants"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[PASS] Chinese settings materialized")
    print(" settings =", output)
    print(" output_root =", output_root)
    print(" input_data =", input_dota_root)
    print(" classes =", classes_path)
    print(" question_similarity_threshold =", threshold)
    print(" max_loop =", official.get("max_loop"))
    print(" agent max_tokens =", cfg["v1_2_0_invariants"]["agent_max_tokens_preserved"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
