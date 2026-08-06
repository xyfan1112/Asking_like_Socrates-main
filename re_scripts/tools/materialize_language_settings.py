#!/usr/bin/env python3
"""Materialize a complete v1.2.3 runtime settings file.

Key policy changes from v1.2.2:
- English is the default newly generated/translated branch.
- Chinese remains a separate branch under the configured Chinese root.
- Dataset selection, classes, all generated paths, SFT topology and evaluation
  topology are resolved together so one RUN_LANG switch changes the full run.
- Chinese class names are never guessed into English.  A deterministic mapping
  template is generated from the real classes.txt and prepare stops until every
  non-English label has been confirmed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VERSION = "1.2.3"
RELEASE_REVISION = "r3"
DIRECT_MODES = (
    "legacy_class_roi_obb",
    "all_image_json_obb",
    "all_image_json_hbb",
    "single_ref_json_obb",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_classes(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    rows = [row for row in rows if row and not row.startswith("#")]
    if not rows:
        raise SystemExit(f"classes.txt is empty: {path}")
    duplicates = sorted({x for x in rows if rows.count(x) > 1})
    if duplicates:
        raise SystemExit(f"classes.txt contains duplicate labels: {duplicates}")
    return rows


def is_ascii_label(value: str) -> bool:
    return bool(value) and all(ord(ch) < 128 for ch in value)


def load_or_create_class_map(
    classes: list[str], source_path: Path, mapping_path: Path, output_root: Path
) -> tuple[list[str], dict[str, Any]]:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source_path)
    template: dict[str, Any] = {
        "schema_version": "class_map_zh_en_v1",
        "source_classes_file": str(source_path),
        "source_sha256": source_hash,
        "instructions": (
            "Keep id and zh unchanged. Fill en with one stable canonical English label. "
            "Do not merge or reorder classes."
        ),
        "classes": [
            {"id": i, "zh": name, "en": name if is_ascii_label(name) else ""}
            for i, name in enumerate(classes)
        ],
    }

    if mapping_path.is_file():
        loaded = json.loads(mapping_path.read_text(encoding="utf-8"))
        old_entries = loaded.get("classes") if isinstance(loaded, dict) else None
        if not isinstance(old_entries, list):
            raise SystemExit(f"Invalid class map schema: {mapping_path}")
        by_id = {int(entry.get("id")): entry for entry in old_entries if isinstance(entry, dict) and "id" in entry}
        for entry in template["classes"]:
            old = by_id.get(entry["id"])
            if old and str(old.get("zh", "")).strip() == entry["zh"]:
                entry["en"] = str(old.get("en", "")).strip()
        # Always rewrite metadata/hash while preserving confirmed translations.
    mapping_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = [entry for entry in template["classes"] if not str(entry.get("en", "")).strip()]
    todo = output_root / "config" / "class_translation_todo.txt"
    todo.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        todo.write_text(
            "\n".join(f"{entry['id']}\t{entry['zh']}\t<fill English>" for entry in missing) + "\n",
            encoding="utf-8",
        )
        print("[BLOCKED] English class translation is incomplete.")
        print(f"  map  = {mapping_path}")
        print(f"  todo = {todo}")
        print("Fill every classes[].en field, then rerun prepare. Class IDs and OBB labels remain unchanged.")
        raise SystemExit(3)

    en = [str(entry["en"]).strip() for entry in template["classes"]]
    if len(set(en)) != len(en):
        dup = sorted({x for x in en if en.count(x) > 1})
        raise SystemExit(f"English class map contains duplicate canonical labels: {dup}")
    todo.unlink(missing_ok=True)
    return en, template


def replace_generated_paths(cfg: dict[str, Any], output_root: Path) -> dict[str, Path]:
    paths = cfg.setdefault("paths", {})
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
    return generated


def model_clone(cfg: dict[str, Any], base_key: str, path: str, served_name: str) -> dict[str, Any]:
    model = copy.deepcopy(cfg.get("models", {}).get(base_key, {}))
    if not model:
        raise KeyError(f"models.{base_key} is required")
    model["path"] = path
    model["served_name"] = served_name
    return model


def derive_grad_accum(training: dict[str, Any], new_world: int, override: str) -> tuple[int, int]:
    per_device = int(training.get("per_device_train_batch_size", 1))
    old_world = int(training.get("world_size", 1))
    old_accum = int(training.get("gradient_accumulation_steps", 1))
    global_batch = per_device * old_world * old_accum
    if override != "auto":
        new_accum = int(override)
    else:
        denom = per_device * new_world
        if global_batch % denom != 0:
            raise SystemExit(
                f"Cannot preserve effective global batch exactly: old={global_batch}, "
                f"per_device={per_device}, new_world={new_world}. Set --sft-gradient-accum explicitly."
            )
        new_accum = global_batch // denom
    if new_accum < 1:
        raise SystemExit("gradient_accumulation_steps must be >= 1")
    return global_batch, new_accum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-settings", required=True, type=Path)
    ap.add_argument("--classes-file", required=True, type=Path)
    ap.add_argument("--lang", choices=("zh", "en"), required=True)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--dataset-variant", choices=("raw", "scene_disjoint"), required=True)
    ap.add_argument("--class-map", type=Path)
    ap.add_argument("--direct-qa-mode", choices=DIRECT_MODES, default="legacy_class_roi_obb")
    ap.add_argument("--eval-gpus", default="0,1,2,3")
    ap.add_argument("--eval-topology", choices=("single", "replica", "replica4"), default="replica")
    ap.add_argument("--eval-base-port", type=int, default=8010)
    ap.add_argument("--eval-gpu-memory-utilization", type=float, default=0.86)
    ap.add_argument("--eval-max-model-len", type=int, default=12288)
    ap.add_argument("--sft-visible-gpus", default="0,1,2,3")
    ap.add_argument("--sft-world-size", type=int, default=4)
    ap.add_argument("--sft-gradient-accum", default="auto")
    ap.add_argument("--data-python", default="")
    ap.add_argument("--sft-python", default="")
    ap.add_argument("--vllm-python", default="")
    ap.add_argument("--require-frozen-vision", action="store_true")
    ap.add_argument("--student-model-family", choices=("rs_eot", "qwen3_omni"), default="rs_eot")
    ap.add_argument("--student-base-model-path", default="")
    ap.add_argument("--student-template", default="")
    ap.add_argument("--student-b1-adapter-output", default="")
    ap.add_argument("--student-b2-adapter-output", default="")
    ap.add_argument("--student-adapter-only", choices=("0", "1"), default="0")
    ap.add_argument("--student-freeze-vision", choices=("0", "1"), default="1")
    ap.add_argument("--student-freeze-projector", choices=("0", "1"), default="0")
    ap.add_argument("--student-lora-target", default="all")
    args = ap.parse_args()

    base_path = args.base_settings.expanduser().resolve()
    classes_path = args.classes_file.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not classes_path.is_file():
        raise FileNotFoundError(classes_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    output = (
        args.output.expanduser().resolve()
        if args.output
        else output_root / "config" / f"settings.{args.lang}.v{VERSION}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    classes = read_classes(classes_path)

    if args.lang == "en":
        map_path = (args.class_map or output_root / "config" / "class_map.zh_en.json").expanduser().resolve()
        active_classes, class_map = load_or_create_class_map(classes, classes_path, map_path, output_root)
        active_classes_file = output_root / "config" / "classes.en.txt"
    else:
        class_map = {
            "schema_version": "identity_zh_v1",
            "source_classes_file": str(classes_path),
            "source_sha256": sha256_file(classes_path),
            "classes": [{"id": i, "zh": name, "en": None} for i, name in enumerate(classes)],
        }
        active_classes = classes
        active_classes_file = output_root / "config" / "classes.zh.txt"
    active_classes_file.parent.mkdir(parents=True, exist_ok=True)
    active_classes_file.write_text("\n".join(active_classes) + "\n", encoding="utf-8")

    cfg = copy.deepcopy(json.loads(base_path.read_text(encoding="utf-8")))
    cfg.setdefault("experiment", {})["name"] = f"v1_2_3_{args.lang}_{args.dataset_variant}"
    cfg["paths"]["dota128_root"] = str(dataset_root)
    generated = replace_generated_paths(cfg, output_root)

    official = cfg.setdefault("official_socratic", {})
    official["input_parquet_dir"] = str(output_root / "pipeline" / "official_socratic" / "input")
    official["raw_output_dir"] = str(output_root / "pipeline" / "official_socratic" / "raw")
    official["postproc_output_dir"] = str(output_root / "pipeline" / "official_socratic" / "postproc")

    cfg["taxonomy"] = {
        "mode": "custom",
        "dataset_name": f"custom_obb_{args.lang}_v{VERSION}",
        "source_classes_file": str(classes_path),
        "classes_file": str(active_classes_file),
        "qa_language": args.lang,
        "strict_exact_final_label": True,
        "preserve_numeric_class_ids": True,
        "class_map": class_map,
    }
    data_conversion = cfg.setdefault("data_conversion", {})
    data_conversion["qa_language"] = args.lang
    data_conversion["direct_qa_mode"] = args.direct_qa_mode
    data_conversion.setdefault("direct_json_max_objects_per_image", 200)
    trajectory = cfg.setdefault("trajectory", {})
    trajectory["qa_language"] = args.lang
    trajectory["prompt_profile"] = "custom_obb_bilingual_v1_2_3"

    runtime = cfg.setdefault("runtime", {})
    workloads = runtime.setdefault("workloads", {})
    for workload, value in (("data", args.data_python), ("sft", args.sft_python), ("vllm", args.vllm_python)):
        if value:
            workloads.setdefault(workload, {})["python"] = str(Path(value).expanduser())
    profiles = runtime.setdefault("gpu_profiles", {})
    profiles["sft_active"] = {
        "visible_devices": args.sft_visible_gpus,
        "world_size": args.sft_world_size,
    }
    profiles["eval_active"] = {
        "topology": args.eval_topology,
        "visible_devices": args.eval_gpus,
        "base_port": args.eval_base_port,
        "replicas": 1 if args.eval_topology == "single" else len([x for x in args.eval_gpus.split(",") if x.strip()]),
        "gpu_memory_utilization": args.eval_gpu_memory_utilization,
        "max_model_len": args.eval_max_model_len,
    }
    # Preserve compatibility with existing scripts that still read legacy profile keys.
    profiles["sft_four_a6000"] = copy.deepcopy(profiles["sft_active"])
    profiles["sft_dual_a6000"] = copy.deepcopy(profiles["sft_active"])
    profiles["eval_replica4"] = copy.deepcopy(profiles["eval_active"])
    profiles["eval_single_gpu"] = {
        "visible_devices": args.eval_gpus.split(",")[0].strip(),
        "gpu_memory_utilization": args.eval_gpu_memory_utilization,
        "max_model_len": args.eval_max_model_len,
    }

    training = cfg.setdefault("training", {})
    global_batch, new_accum = derive_grad_accum(training, args.sft_world_size, args.sft_gradient_accum)
    training["world_size"] = args.sft_world_size
    training["gradient_accumulation_steps"] = new_accum
    training["b1_adapter_output"] = str(output_root / "training" / "b1_matched_lora")
    training["b2_adapter_output"] = str(output_root / "training" / "b2_socratic_lora")
    training["b1_merged_output"] = str(output_root / "models" / "RS-EoT-Custom-B1-Matched-Merged")
    training["b2_merged_output"] = str(output_root / "models" / "RS-EoT-Custom-B2-Socratic-Merged")
    training["b1_direct_standalone_adapter_output"] = str(output_root / "training" / "b1_direct_standalone_lora")
    training["b1_direct_standalone_merged_output"] = str(output_root / "models" / "RS-EoT-B1-Direct-Standalone-Merged")
    training["d1_adapter_output"] = str(output_root / "training" / "d1_direct_only_lora")
    training["d1_merged_output"] = str(output_root / "models" / "RS-EoT-D1-Direct-Only-Merged")

    models = cfg.setdefault("models", {})
    # Always keep the original RS-EoT route intact.
    rs_base_key = "rs_eot"
    if rs_base_key not in models:
        raise KeyError("models.rs_eot is required")
    models["rs_eot_b1_direct"] = model_clone(
        cfg, rs_base_key, training["b1_direct_standalone_merged_output"], "rs-eot-b1-direct"
    )
    models["rs-eot-b1-direct"] = copy.deepcopy(models["rs_eot_b1_direct"])
    models["rs_eot_b2_socratic"] = model_clone(
        cfg, rs_base_key, training["b2_merged_output"], "rs-eot-b2-socratic"
    )

    if args.student_model_family == "qwen3_omni":
        base_key = "qwen3_omni_b0"
        base_model_path = str(Path(args.student_base_model_path).expanduser())
        if not base_model_path:
            raise SystemExit("--student-base-model-path is required for qwen3_omni")
        base_model = copy.deepcopy(models.get(base_key, {}))
        if not base_model:
            raise KeyError("models.qwen3_omni_b0 is required in base settings")
        base_model.update({
            "path": base_model_path,
            "served_name": "qwen3-omni-b0",
            "architecture": "qwen3_omni",
            "training_template": args.student_template or "qwen3_omni",
            "llama_factory_family": "qwen3",
            "runtime_workloads": {"vllm": "qwen3_vllm", "sft": "qwen3_sft"},
            "adapter_only": False,
        })
        models[base_key] = base_model
        b1_adapter = str(Path(args.student_b1_adapter_output).expanduser())
        b2_adapter = str(Path(args.student_b2_adapter_output).expanduser())
        if not b1_adapter or not b2_adapter:
            raise SystemExit("Qwen3-Omni adapter output paths are required")
        b1_model = model_clone(cfg, base_key, base_model_path, "qwen3-omni-b1-direct")
        b1_model.update({
            "base_model_key": base_key,
            "base_served_name": "qwen3-omni-b0",
            "adapter_path": b1_adapter,
            "adapter_name": "qwen3-omni-b1-direct",
            "adapter_only": True,
        })
        b2_model = model_clone(cfg, base_key, base_model_path, "qwen3-omni-b2-socratic")
        b2_model.update({
            "base_model_key": base_key,
            "base_served_name": "qwen3-omni-b0",
            "adapter_path": b2_adapter,
            "adapter_name": "qwen3-omni-b2-socratic",
            "adapter_only": True,
        })
        models["qwen3_omni_b1_direct"] = b1_model
        models["qwen3_omni_b2_socratic"] = b2_model
        training["base_model_key"] = base_key
        training["template"] = args.student_template or "qwen3_omni"
        training["adapter_only"] = args.student_adapter_only == "1"
        training["freeze_vision_tower"] = args.student_freeze_vision == "1"
        training["freeze_multi_modal_projector"] = args.student_freeze_projector == "1"
        training["lora_target"] = args.student_lora_target
        training["b1_direct_standalone_adapter_output"] = b1_adapter
        training["b2_adapter_output"] = b2_adapter
        routing = {
            "student_family": "qwen3_omni",
            "b0_model_key": base_key,
            "b1_model_key": "qwen3_omni_b1_direct",
            "b2_model_key": "qwen3_omni_b2_socratic",
            "adapter_only": True,
        }
    else:
        training["base_model_key"] = "rs_eot"
        training["adapter_only"] = False
        routing = {
            "student_family": "rs_eot",
            "b0_model_key": "rs_eot",
            "b1_model_key": "rs_eot_b1_direct",
            "b2_model_key": "rs_eot_b2_socratic",
            "adapter_only": False,
        }

    if args.require_frozen_vision and not bool(training.get("freeze_vision_tower", True)):
        raise SystemExit("freeze_vision_tower=false but v1.2.3 requires the vision tower to remain frozen")
    cfg["model_routing"] = routing

    signature = {
        "schema_version": "v1_2_3_runtime_signature",
        "version": VERSION,
        "release_revision": RELEASE_REVISION,
        "base_settings": str(base_path),
        "base_settings_sha256": sha256_file(base_path),
        "language": args.lang,
        "dataset_variant": args.dataset_variant,
        "dataset_root": str(dataset_root),
        "source_classes_file": str(classes_path),
        "source_classes_sha256": sha256_file(classes_path),
        "active_classes_file": str(active_classes_file),
        "active_classes_sha256": sha256_file(active_classes_file),
        "class_count": len(active_classes),
        "direct_qa_mode": args.direct_qa_mode,
        "sft": {
            "visible_gpus": args.sft_visible_gpus,
            "world_size": args.sft_world_size,
            "gradient_accumulation_steps": new_accum,
            "effective_global_batch": global_batch,
            "freeze_vision_tower": bool(training.get("freeze_vision_tower", True)),
            "freeze_multi_modal_projector": bool(training.get("freeze_multi_modal_projector", False)),
            "finetuning_type": training.get("finetuning_type", "lora"),
        },
        "evaluation": profiles["eval_active"],
        "student": cfg["model_routing"],
        "scientific_controls_preserved": {
            "question_similarity_threshold": trajectory.get("question_similarity_threshold"),
            "official_max_loop": official.get("max_loop"),
            "agent_max_tokens": {
                role: cfg.get("agents", {}).get(role, {}).get("max_tokens")
                for role in ("reasoner", "perceiver", "verifier")
            },
        },
    }
    cfg["v1_2_3"] = signature

    for path in [output_root, output.parent, *generated.values()]:
        Path(path).mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(signature, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[PASS] v1.2.3-{RELEASE_REVISION} settings materialized")
    print(f"  settings       = {output}")
    print(f"  language       = {args.lang}")
    print(f"  dataset        = {dataset_root} ({args.dataset_variant})")
    print(f"  source classes = {classes_path} ({len(classes)})")
    print(f"  active classes = {active_classes_file}")
    print(f"  output root    = {output_root}")
    print(f"  Direct mode    = {args.direct_qa_mode}")
    print(f"  SFT            = GPUs {args.sft_visible_gpus}; world={args.sft_world_size}; accum={new_accum}; global_batch={global_batch}")
    print(f"  Eval           = {args.eval_topology} GPUs {args.eval_gpus}; ports {args.eval_base_port}..{args.eval_base_port + profiles['eval_active']['replicas'] - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
