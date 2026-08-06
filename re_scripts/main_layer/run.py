#!/usr/bin/env python3
"""Unified experiment entry point for custom OBB bilingual patch v1.2.3.

The main layer is intentionally thin: it materializes a base settings file plus
one experiment profile, chooses the correct workload Python, records the exact
runtime settings, then dispatches to data/train/test scripts. Scientific logic
remains in the three domain layers.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASE_REVISION = "1.2.3-r3"
sys.path.insert(0, str(ROOT / "main_layer"))
from config import load_json, materialize_settings, resolve_python  # noqa: E402
from taxonomy import materialize_runtime_taxonomy  # noqa: E402

PY_COMMANDS: dict[str, tuple[str, str]] = {
    "preflight": ("data", "main_layer/00_preflight.py"),
    "runtime-check": ("data", "main_layer/runtime.py"),
    "audit-data": ("data", "data_layer/01_audit_dota128.py"),
    "split-scenes": ("data", "tools/build_scene_disjoint_dota128.py"),
    "build-ref": ("data", "data_layer/02_build_dota128_ref.py"),
    "validate-ref": ("data", "data_layer/03_validate_dota128_ref.py"),
    "visualize-ref": ("data", "data_layer/04_visualize_dota128_ref.py"),
    "build-agent-input": ("data", "data_layer/05_build_agent_inputs.py"),
    "audit-lineage": ("data", "data_layer/05c_audit_lineage.py"),
    "build-official-parquet": ("data", "data_layer/official_socratic/01_build_official_parquet.py"),
    "generate-custom-trajectories": ("data", "data_layer/06_generate_trajectories.py"),
    "filter-custom-trajectories": ("data", "data_layer/07_filter_trajectories.py"),
    "build-custom-llamafactory": ("data", "data_layer/08_build_dota128_llamafactory.py"),
    "convert-official-llamafactory": ("data", "data_layer/official_socratic/04_convert_official_to_llamafactory.py"),
    "prepare-direct-sft": ("sft", "train_layer/00_prepare_direct_only.py"),
    "validate-train": ("sft", "train_layer/01_validate_training_data.py"),
    "generate-train-config": ("sft", "train_layer/02_generate_configs.py"),
    "validate-lf-contract": ("sft", "train_layer/02a_validate_llamafactory_contract.py"),
    "generate-official-train-config": ("sft", "train_layer/02b_generate_official_llamafactory_configs.py"),
    "build-rl-data": ("data", "train_layer/rl/01_build_grounding_rl_data.py"),
    "rl-preflight": ("rl", "train_layer/rl/00_preflight_rl.py"),
    "score-vrsbench": ("data", "test_layer/07_score_existing_vrsbench.py"),
    "compare-b0-b1-b2": ("data", "test_layer/13_compare_b0_b1_b2.py"),
    "check-taxonomy": ("data", "tools/check_custom_taxonomy.py"),
    "check-debug": ("data", "tools/check_latest_debug_audit.py"),
    "check-debug-infra": ("data", "tools/check_debug_infrastructure.py"),
    "inspect-api-truncation": ("data", "tools/inspect_api_truncation.py"),
    "generate-d1-config": ("sft", "train_layer/02c_generate_d1_direct_only_config.py"),
    "validate-d1": ("sft", "train_layer/01c_validate_d1_direct_only.py"),
    "register-d1-model": ("data", "tools/register_d1_model.py"),
    "validate-b1-direct-standalone": ("sft", "train_layer/01d_validate_b1_direct_standalone.py"),
    "generate-b1-direct-config": ("sft", "train_layer/02d_generate_b1_direct_standalone_config.py"),
    "validate-b1-direct-contract": ("sft", "train_layer/02e_validate_b1_direct_contract.py"),
    "register-b1-direct-model": ("data", "tools/register_b1_direct_model.py"),
    "register-b2-model": ("data", "tools/register_b2_model.py"),
    "omni-preflight": ("vllm", "tools/check_qwen3_omni_4gpu.py"),
    "materialize-shared-omni-agents": ("data", "tools/materialize_shared_omni_agent_settings.py"),
    "check-shared-omni-endpoint": ("data", "tools/check_shared_omni_endpoint.py"),
}

SHELL_COMMANDS = {
    "data-full": "main_layer/run_data_full.sh",
    "b1-from-scratch": "main_layer/run_b1_from_scratch.sh",
    "start-agents": "data_layer/agents/start_three_agents.sh",
    "stop-agents": "data_layer/agents/stop_three_agents.sh",
    "reset-official": "data_layer/official_socratic/00_reset_official_outputs.sh",
    "official-generate": "data_layer/official_socratic/02_run_official_generation.sh",
    "official-postproc": "data_layer/official_socratic/03_run_official_postproc.sh",
    "official-socratic-full": "main_layer/run_official_socratic_full.sh",
    "resume-official-artifacts": "main_layer/resume_official_artifacts.sh",
    "prepare-custom-sft": "main_layer/run_training_prepare.sh",
    "train-b1": "train_layer/03_train_b1_direct.sh",
    "train-b2": "train_layer/04_train_b2_socratic.sh",
    "train-official": "train_layer/03b_train_official_llamafactory.sh",
    "merge": "train_layer/05_export_merge.sh",
    "start-eval": "test_layer/start_eval_model.sh",
    "stop-eval": "test_layer/stop_eval_model.sh",
    "test-matrix": "main_layer/run_test_matrix.sh",
    "test-matrix-replica4": "main_layer/run_test_matrix_replica4.sh",
    "rl-stage1": "train_layer/rl/03_launch_stage1_grounding.sh",
    "train-d1": "train_layer/03c_train_d1_direct_only.sh",
    "merge-d1": "train_layer/05c_export_d1_direct_only.sh",
    "train-b1-direct-standalone": "train_layer/03d_train_b1_direct_standalone.sh",
    "merge-b1-direct-standalone": "train_layer/05d_export_b1_direct_standalone.sh",
}

INFRA_COMMANDS = {"start-agents", "stop-agents", "start-eval", "stop-eval"}




def _run_checked(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    code = subprocess.call(cmd)
    if code != 0:
        raise SystemExit(code)


def _ensure_agent_inputs(settings: dict[str, Any], materialized: Path) -> None:
    """Build missing prerequisites before the official parquet conversion.

    The previous v4 runner assumed data-full had reached step 6.  In warn/strict
    audit scenarios that assumption was easy to miss, so the public command now
    repairs the dependency chain explicitly.
    """
    work = Path(settings["paths"]["pipeline_work_root"]) / "agent_inputs"
    train_input = work / "train_agent_inputs.jsonl"
    val_input = work / "val_agent_inputs.jsonl"
    if train_input.is_file() and val_input.is_file():
        return
    python_bin = resolve_python(settings, "data")
    print("[AUTO] agent_inputs 缺失，自动执行数据前置步骤：audit → build-ref → validate-ref → build-agent-input")
    for rel in (
        "data_layer/01_audit_dota128.py",
        "data_layer/02_build_dota128_ref.py",
        "data_layer/03_validate_dota128_ref.py",
        "data_layer/05_build_agent_inputs.py",
    ):
        _run_checked([python_bin, str(ROOT / rel), "--settings", str(materialized)])
    if not train_input.is_file():
        raise FileNotFoundError(
            f"自动准备后仍缺少 {train_input}。请查看 "
            f"{work / 'build_report.json'} 和 data_layer/README_数据层.md"
        )

def main() -> None:
    print(f"[MAIN RUNNER] release={RELEASE_REVISION} file={Path(__file__).resolve()}")
    choices = sorted(set(PY_COMMANDS) | set(SHELL_COMMANDS) | {"show-config"})
    ap = argparse.ArgumentParser(
        description="OBB × Asking Like Socrates experiment runner"
    )
    ap.add_argument("command", choices=choices)
    ap.add_argument("--settings", default=str(ROOT / "settings.json"))
    ap.add_argument("--profile", help="Experiment profile name or JSON path")
    ap.add_argument(
        "--lang",
        choices=("zh", "en"),
        help="QA/Reasoner/Perceiver/Verifier natural language; structural tags stay unchanged",
    )
    ap.add_argument(
        "--classes-file",
        "--classes",
        dest="classes_file",
        help="UTF-8 classes.txt; one canonical class per line, line order equals class_id",
    )
    args, rest = ap.parse_known_args()

    materialized = materialize_settings(args.settings, args.profile)
    settings = load_json(materialized)
    catalog = None

    if args.command in INFRA_COMMANDS:
        if args.lang or args.classes_file:
            print(
                f"[INFO] {args.command} only manages a model server; "
                "--lang/--classes-file are ignored. Use a zh settings file "
                "to place logs/PIDs under the zh output root."
            )
        os.environ["ALS_SETTINGS_PATH"] = str(materialized)
        print(f"[CONFIG] materialized settings: {materialized}")
        print(f"[INFRA] language-neutral command: {args.command}")
    else:
        catalog = materialize_runtime_taxonomy(
            settings,
            classes_file=args.classes_file,
            language=args.lang,
        )
        official = settings.setdefault("official_socratic", {})
        if catalog.language == "zh":
            official["image_meta_pre"] = (
                "所提供图像为遥感 RGB 图像。只能依据可见图像证据回答。"
                "坐标必须遵守原始任务中明确给出的坐标制。\n"
            )
            official["verify_inst"] = (
                "最终类别必须与当前 classes.txt 中某一标准类别完全一致。"
                "分类任务的中间视觉问题必须类别中性，不得直接说出、比较或猜测任何候选类别。"
                "定位任务必须始终指向同一目标和正确图像区域。"
                "启用 GT teacher forcing 时，轨迹中的近似坐标不要求与 GT 数值完全相同，"
                "但错误目标、错误类别、错误区域或非法四边形必须拒绝。"
            )
        else:
            official["image_meta_pre"] = (
                "The provided imagery is a remote-sensing RGB image. Answer only "
                "from visible image evidence and follow the coordinate convention "
                "stated in the original task.\n"
            )
            official["verify_inst"] = (
                "The final class must exactly match one canonical label from the "
                "active classes.txt. Intermediate classification questions must be "
                "class-neutral and must not reveal, compare, or guess candidate labels. "
                "Grounding must stay bound to the same target and image region. With "
                "GT teacher forcing, approximate trajectory coordinates need not equal "
                "GT numerically, but wrong targets, classes, regions, or invalid "
                "quadrilaterals must be rejected."
            )
        materialized.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.environ["ALS_QA_LANG"] = catalog.language
        if catalog.classes_file:
            os.environ["ALS_CLASSES_FILE"] = catalog.classes_file
        else:
            os.environ.pop("ALS_CLASSES_FILE", None)
        os.environ["ALS_TAXONOMY_SHA256"] = catalog.sha256
        os.environ["ALS_SETTINGS_PATH"] = str(materialized)
        print(f"[CONFIG] materialized settings: {materialized}")
        print(
            f"[TAXONOMY] mode={'custom' if catalog.custom else 'dota'} "
            f"classes={len(catalog.labels)} lang={catalog.language} "
            f"classes_file={catalog.classes_file or '(built-in DOTA fallback)'} "
            f"sha256={catalog.sha256[:16]}"
        )
    if args.command == "show-config":
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return

    if args.command == "build-official-parquet":
        _ensure_agent_inputs(settings, materialized)

    if args.command in PY_COMMANDS:
        workload, rel = PY_COMMANDS[args.command]
        python_bin = resolve_python(settings, workload)
        cmd = [python_bin, str(ROOT / rel), "--settings", str(materialized), *rest]
    else:
        cmd = ["bash", str(ROOT / SHELL_COMMANDS[args.command]), str(materialized), *rest]
    print("+", " ".join(str(x) for x in cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
