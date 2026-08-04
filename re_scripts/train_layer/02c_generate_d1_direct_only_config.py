#!/usr/bin/env python3
"""Generate isolated D1 Direct-only train/export YAML files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402
from config import resolve_training_template  # noqa: E402


def method_block(training: dict) -> str:
    mode = training.get("finetuning_type", "lora")
    if mode != "lora":
        raise ValueError("D1 v1.2.2 supports the stable LoRA configuration only")
    return "\n".join([
        "stage: sft",
        "do_train: true",
        "do_eval: true",
        "finetuning_type: lora",
        "lora_target: all",
        f"lora_rank: {training['lora_rank']}",
        f"lora_alpha: {training['lora_alpha']}",
        f"lora_dropout: {training['lora_dropout']}",
        f"freeze_vision_tower: {str(training['freeze_vision_tower']).lower()}",
        f"freeze_multi_modal_projector: {str(training['freeze_multi_modal_projector']).lower()}",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    training = settings["training"]
    base_key = training["base_model_key"]
    base = settings["models"][base_key]["path"]
    template = resolve_training_template(settings, base_key)
    data = settings["paths"]["dota128_llamafactory_root"]
    output = training["d1_adapter_output"]
    merged = training["d1_merged_output"]
    run = Path(settings["paths"]["training_run_root"])
    cfg_dir = run / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    train_yaml = f"""### model
model_name_or_path: {base}
image_max_pixels: {training.get('image_max_pixels', 802816)}
trust_remote_code: true

### method
{method_block(training)}

### dataset
dataset_dir: {data}
dataset: dota128_direct_train_official
eval_dataset: dota128_direct_val_official
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
    export_yaml = f"""model_name_or_path: {base}
adapter_name_or_path: {output}
template: {template}
finetuning_type: lora
export_dir: {merged}
export_size: 5
export_device: cpu
export_legacy_format: false
trust_remote_code: true
"""
    train_path = cfg_dir / "d1_direct_only_lora.yaml"
    export_path = cfg_dir / "d1_export.yaml"
    train_path.write_text(train_yaml, encoding="utf-8")
    export_path.write_text(export_yaml, encoding="utf-8")
    print("[D1 CONFIG] PASS")
    print(" train =", train_path)
    print(" export =", export_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
