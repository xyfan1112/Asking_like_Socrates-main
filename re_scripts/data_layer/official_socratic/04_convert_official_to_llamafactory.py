#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main_layer"))
from common import load_settings, read_jsonl, settings_from_cli  # noqa: E402
from config import resolve_llama_factory_dir, resolve_training_template  # noqa: E402

LEADING_THINK_RE = re.compile(r"^\s*<think>\s*", re.IGNORECASE)


def resolve_image(item: Dict[str, Any]) -> str:
    image_list = item.get("image") or []
    roots = item.get("image_root") or {}
    if not image_list:
        raise ValueError("Missing image list")
    pair = image_list[0]
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"Invalid image pair: {pair!r}")
    modality, filename = pair
    path = Path(roots.get(modality, "")) / str(filename)
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def normalize_assistant_content(full_trace: str, template: str) -> str:
    text = str(full_trace or "").strip()
    if template == "qwen2_vl_thinking":
        # This LLaMA-Factory template injects the opening '<think>' token in the
        # assistant prefix. Keep the generated closing tag and final answer, but
        # remove exactly one leading opening tag to avoid nested <think> blocks.
        text = LEADING_THINK_RE.sub("", text, count=1)
    if "</think>" not in text:
        raise ValueError("Converted assistant response does not contain </think>")
    return text


def replace_final_with_gt(full_trace: str, gt: Any, enabled: bool) -> tuple[str, str]:
    text = str(full_trace or "").strip()
    generated = text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else ""
    if not enabled:
        return text, generated
    gt_text = str(gt or "").strip()
    if not gt_text:
        raise ValueError("teacher_force_final_answer is enabled but raw_gt is empty")
    if "</think>" not in text:
        raise ValueError("Cannot teacher-force a trace without </think>")
    prefix = text.rsplit("</think>", 1)[0]
    return f"{prefix}</think>{gt_text}", generated


def dataset_entry(file_name: str) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def final_only_content(gt: Any, template: str) -> str:
    answer = str(gt or "").strip()
    if not answer:
        raise ValueError("final-only counterpart has empty GT")
    return f"</think>\n{answer}" if template == "qwen2_vl_thinking" else answer


def final_only_row(
    *,
    pair_id: str,
    query: str,
    gt: Any,
    image_path: str,
    task: str | None,
    template: str,
    split: str,
) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": str(query).strip()},
            {"role": "assistant", "content": final_only_content(gt, template)},
        ],
        "images": [image_path],
        "metadata": {
            "id": pair_id,
            "comparison_pair_id": pair_id,
            "task": task,
            "split": split,
            "raw_gt": str(gt or "").strip(),
            "trajectory_representation": "final_only",
            "teacher_forced_final": True,
            "official_socraticagent": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official SocraticAgent postproc output to LLaMA-Factory multimodal ShareGPT format.")
    parser.add_argument("--settings", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--register", action="store_true", help="Patch the official LLaMA-Factory/data/dataset_info.json after creating files.")
    args = parser.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    official = settings["official_socratic"]
    training = settings["training"]
    base_key = training["base_model_key"]
    lf_root = resolve_llama_factory_dir(settings, base_key)
    template = resolve_training_template(settings, base_key)
    out_root = Path(settings["paths"]["dota128_llamafactory_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input) if args.input else Path(official["postproc_output_dir"]) / f"dota128_{args.split}_official_merge.json"
    items = json.loads(input_path.read_text(encoding="utf-8"))
    source_task_counts = Counter(str(item.get("task", "unknown")) for item in items)
    if args.split == "train":
        min_g = int(settings["trajectory"].get("min_strict_grounding_samples", 20))
        min_c = int(settings["trajectory"].get("min_strict_classification_samples", 20))
        if source_task_counts.get("ref_grounding_obb", 0) < min_g or source_task_counts.get("ref_classification", 0) < min_c:
            raise RuntimeError(
                f"Strict postproc data is too small/imbalanced: {dict(source_task_counts)}; "
                f"required grounding/classification >= {min_g}/{min_c}."
            )
    converted: List[Dict[str, Any]] = []
    matched_final_only: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        try:
            forced_trace, generated_answer = replace_final_with_gt(
                item.get("thinking", ""),
                item.get("raw_gt"),
                bool(settings["trajectory"].get("teacher_force_final_answer", True)),
            )
            image_path = resolve_image(item)
            pair_id = str(item.get("id") or "")
            if not pair_id:
                raise ValueError("Missing accepted trajectory id")
            query = str(item.get("query") or item.get("raw_query") or "").strip()
            task = item.get("task")
            converted.append({
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": normalize_assistant_content(forced_trace, template)},
                ],
                "images": [image_path],
                "metadata": {
                    "id": pair_id,
                    "comparison_pair_id": pair_id,
                    "task": task,
                    "data_source": item.get("data_source"),
                    "raw_gt": item.get("raw_gt"),
                    "generated_answer_before_teacher_force": generated_answer,
                    "teacher_forced_final": bool(settings["trajectory"].get("teacher_force_final_answer", True)),
                    "official_socraticagent": True,
                    "trajectory_representation": "socratic",
                },
            })
            matched_final_only.append(
                final_only_row(
                    pair_id=pair_id,
                    query=query,
                    gt=item.get("raw_gt"),
                    image_path=image_path,
                    task=task,
                    template=template,
                    split=args.split,
                )
            )
        except Exception as exc:
            errors.append({"index": idx, "id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"})

    if errors:
        error_path = out_root / f"dota128_{args.split}_official_conversion_errors.json"
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"Failed to convert {len(errors)} official postproc rows. See {error_path}")
    if not converted:
        raise RuntimeError(f"No valid official SocraticAgent rows found in {input_path}")

    soc_name = f"dota128_socratic_{args.split}_official.json"
    soc_path = out_root / soc_name
    soc_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")

    # Reuse direct supervision generated by the shared preprocessing layer.
    agent_dir = Path(settings["paths"]["pipeline_work_root"]) / "agent_inputs"
    direct_source = agent_dir / f"{args.split}_direct.json"
    direct_name = f"dota128_direct_{args.split}_official.json"
    direct_path = out_root / direct_name
    if direct_source.exists():
        shutil.copyfile(direct_source, direct_path)
    elif args.split == "train":
        raise FileNotFoundError(f"Missing direct supervision file: {direct_source}")

    report: Dict[str, Any] = {
        "input": str(input_path),
        "socratic_output": str(soc_path),
        "socratic_samples": len(converted),
        "socratic_task_counts": dict(source_task_counts),
        "template": template,
        "leading_think_removed": template == "qwen2_vl_thinking",
        "teacher_force_final_answer": bool(settings["trajectory"].get("teacher_force_final_answer", True)),
        "matched_final_only_samples": len(matched_final_only),
    }

    if args.split == "train" and direct_path.exists():
        direct_rows = json.loads(direct_path.read_text(encoding="utf-8"))
        b1_name = "dota128_b1_matched_train_official.json"
        b2_name = "dota128_b2_matched_train_official.json"
        b1_path = out_root / b1_name
        b2_path = out_root / b2_name
        b1_rows = direct_rows + matched_final_only
        b2_rows = direct_rows + converted
        b1_path.write_text(json.dumps(b1_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        b2_path.write_text(json.dumps(b2_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        mixed_name = "dota128_mixed_train_official.json"
        mixed_path = out_root / mixed_name
        mixed_path.write_text(json.dumps(b2_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        val_source = agent_dir / "val_direct.json"
        val_path = out_root / "dota128_direct_val_official.json"
        if val_source.exists():
            shutil.copyfile(val_source, val_path)
        ref_val_path = out_root / "dota128_ref_val_official.json"
        val_agent_path = agent_dir / "val_agent_inputs.jsonl"
        val_rows = []
        if val_agent_path.is_file():
            for val_item in read_jsonl(val_agent_path):
                val_rows.append(
                    final_only_row(
                        pair_id=str(val_item["id"]),
                        query=str(val_item["query"]),
                        gt=val_item["gt"],
                        image_path=str(Path(val_item["image_path"]).expanduser().resolve()),
                        task=val_item.get("task"),
                        template=template,
                        split="val",
                    )
                )
        if not val_rows:
            raise RuntimeError(f"Matched Ref validation data is empty: {val_agent_path}")
        ref_val_path.write_text(
            json.dumps(val_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report.update({
            "direct_output": str(direct_path),
            "direct_samples": len(direct_rows),
            "mixed_output": str(mixed_path),
            "mixed_samples": len(direct_rows) + len(converted),
            "direct_val_output": str(val_path) if val_path.exists() else None,
            "b1_matched_output": str(b1_path),
            "b2_matched_output": str(b2_path),
            "b1_matched_samples": len(b1_rows),
            "b2_matched_samples": len(b2_rows),
            "matched_pair_count": len(converted),
            "matched_row_count_equal": len(b1_rows) == len(b2_rows),
            "ref_val_output": str(ref_val_path),
            "ref_val_samples": len(val_rows),
        })

    # Authoritative registry must live in dataset_dir because LLaMA-Factory
    # always reads <dataset_dir>/dataset_info.json.
    local_info_path = out_root / "dataset_info.json"
    local_info = {}
    if local_info_path.exists():
        local_info = json.loads(local_info_path.read_text(encoding="utf-8"))
    registry_files = {
        "dota128_socratic_train_official": out_root / "dota128_socratic_train_official.json",
        "dota128_direct_train_official": out_root / "dota128_direct_train_official.json",
        "dota128_mixed_train_official": out_root / "dota128_mixed_train_official.json",
        "dota128_b1_matched_train_official": out_root / "dota128_b1_matched_train_official.json",
        "dota128_b2_matched_train_official": out_root / "dota128_b2_matched_train_official.json",
        "dota128_ref_val_official": out_root / "dota128_ref_val_official.json",
    }
    for registry_name, registry_path in registry_files.items():
        if registry_path.is_file():
            local_info[registry_name] = dataset_entry(registry_path.name)
    val_file = out_root / "dota128_direct_val_official.json"
    if val_file.exists():
        local_info["dota128_direct_val_official"] = dataset_entry(val_file.name)
    local_info_path.write_text(json.dumps(local_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["authoritative_dataset_info"] = str(local_info_path)

    if args.register:
        info_path = lf_root / "data" / "dataset_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for registry_name, registry_path in registry_files.items():
            if registry_path.is_file():
                info[registry_name] = dataset_entry(str(registry_path))
        if val_file.exists():
            info["dota128_direct_val_official"] = dataset_entry(str(val_file))
        backup = info_path.with_name("dataset_info.json.re_scripts_v4_3_1_backup")
        if not backup.exists():
            shutil.copyfile(info_path, backup)
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["official_registry_mirror"] = str(info_path)
        report["official_registry_backup"] = str(backup)

    report_path = out_root / f"dota128_{args.split}_official_llamafactory_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
