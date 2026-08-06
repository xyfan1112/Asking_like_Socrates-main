#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402
from config import resolve_training_template  # noqa: E402


def method_block(training: dict) -> str:
    mode = training.get("finetuning_type", "lora")
    lines = ["stage: sft", "do_train: true", "do_eval: true", f"finetuning_type: {mode}"]
    if mode == "lora":
        lines += [
            "lora_target: all",
            f"lora_rank: {training['lora_rank']}",
            f"lora_alpha: {training['lora_alpha']}",
            f"lora_dropout: {training['lora_dropout']}",
        ]
    elif mode != "full":
        raise ValueError(f"Unsupported finetuning_type: {mode}")
    lines += [
        f"freeze_vision_tower: {str(training['freeze_vision_tower']).lower()}",
        f"freeze_multi_modal_projector: {str(training['freeze_multi_modal_projector']).lower()}",
    ]
    return "\n".join(lines)


def train_yaml(settings: dict, dataset: str, eval_dataset: str, output: str) -> str:
    training = settings["training"]
    base_key = training["base_model_key"]
    base = settings["models"][base_key]["path"]
    template = resolve_training_template(settings, base_key)
    data = settings["paths"]["dota128_llamafactory_root"]
    return f"""### model
model_name_or_path: {base}
image_max_pixels: {training.get('image_max_pixels', 262144)}
trust_remote_code: true

### method
{method_block(training)}

### dataset
dataset_dir: {data}
dataset: {dataset}
eval_dataset: {eval_dataset}
template: {template}
cutoff_len: {training['cutoff_len']}
overwrite_cache: {str(training.get('overwrite_cache', True)).lower()}
preprocessing_num_workers: 8
dataloader_num_workers: 4

### output
output_dir: {output}
logging_steps: 1
save_strategy: epoch
eval_strategy: epoch
plot_loss: true
overwrite_output_dir: false
save_total_limit: 2
report_to: none

### train
per_device_train_batch_size: {training['per_device_train_batch_size']}
per_device_eval_batch_size: {training['per_device_eval_batch_size']}
gradient_accumulation_steps: {training['gradient_accumulation_steps']}
learning_rate: {training['learning_rate']}
num_train_epochs: {training['epochs']}
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: {str(training['bf16']).lower()}
gradient_checkpointing: {str(training['gradient_checkpointing']).lower()}
optim: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0
ddp_find_unused_parameters: false
seed: {settings['project']['seed']}
"""


def export_yaml(settings: dict, adapter: str, merged: str) -> str:
    training = settings["training"]
    if training.get("finetuning_type", "lora") != "lora":
        raise ValueError("Export merge is only needed for LoRA runs")
    base_key = training["base_model_key"]
    base = settings["models"][base_key]["path"]
    template = resolve_training_template(settings, base_key)
    return f"""model_name_or_path: {base}
adapter_name_or_path: {adapter}
template: {template}
finetuning_type: lora
export_dir: {merged}
export_size: 5
export_device: cpu
export_legacy_format: false
trust_remote_code: true
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    out = Path(settings["paths"]["training_run_root"]) / "configs"
    out.mkdir(parents=True, exist_ok=True)
    training = settings["training"]
    template = resolve_training_template(settings)

    data_root = Path(settings["paths"]["dota128_llamafactory_root"])
    dataset_info = data_root / "dataset_info.json"
    if not dataset_info.is_file():
        raise FileNotFoundError(
            f"Missing {dataset_info}. Run: python main_layer/run.py prepare-direct-sft "
            f"--settings settings.json --register"
        )
    b1_dataset = "dota128_b1_matched_train_official"
    b2_dataset = "dota128_b2_matched_train_official"
    eval_dataset = "dota128_ref_val_official"

    (out / "b1_direct_lora.yaml").write_text(
        train_yaml(settings, b1_dataset, eval_dataset, training["b1_adapter_output"]),
        encoding="utf-8",
    )
    (out / "b2_socratic_lora.yaml").write_text(
        train_yaml(settings, b2_dataset, eval_dataset, training["b2_adapter_output"]),
        encoding="utf-8",
    )
    if training.get("finetuning_type", "lora") == "lora":
        (out / "b1_export.yaml").write_text(
            export_yaml(settings, training["b1_adapter_output"], training["b1_merged_output"]),
            encoding="utf-8",
        )
        (out / "b2_export.yaml").write_text(
            export_yaml(settings, training["b2_adapter_output"], training["b2_merged_output"]),
            encoding="utf-8",
        )
    report = {
        "config_dir": str(out),
        "base_model_key": training["base_model_key"],
        "template_resolved": template,
        "loss_mode": training.get("loss_mode", "standard_causal_lm"),
        "b1_dataset": b1_dataset,
        "b2_dataset": b2_dataset,
        "eval_dataset": eval_dataset,
        "note": (
            "Controlled comparison: B1=the same Direct rows plus final-only Ref pairs; "
            "B2=the same Direct rows plus Socratic traces for exactly those Ref pair IDs. "
            "Questions, images, GT finals, and row counts are matched."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
