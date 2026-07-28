#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402
from config import resolve_training_template  # noqa: E402


def render(settings: dict, dataset_name: str, output_dir: str) -> str:
    training = settings["training"]
    base_key = training["base_model_key"]
    base = settings["models"][base_key]["path"]
    template = resolve_training_template(settings, base_key)
    dataset_dir = settings["paths"]["dota128_llamafactory_root"]
    full = training.get("finetuning_type", "lora") == "full"

    method_lines = [
        "stage: sft",
        "do_train: true",
        f"finetuning_type: {training.get('finetuning_type', 'lora')}",
    ]
    if full:
        method_lines += [
            f"freeze_vision_tower: {str(training.get('freeze_vision_tower', False)).lower()}",
            f"freeze_multi_modal_projector: {str(training.get('freeze_multi_modal_projector', False)).lower()}",
            "freeze_language_model: false",
            "deepspeed: examples/deepspeed/ds_z3_config.json",
        ]
    else:
        method_lines += [
            "lora_target: all",
            f"lora_rank: {training['lora_rank']}",
            f"lora_alpha: {training['lora_alpha']}",
            f"lora_dropout: {training['lora_dropout']}",
            f"freeze_vision_tower: {str(training.get('freeze_vision_tower', True)).lower()}",
            f"freeze_multi_modal_projector: {str(training.get('freeze_multi_modal_projector', False)).lower()}",
        ]

    return f"""### model
model_name_or_path: {base}
image_max_pixels: {training.get('image_max_pixels', 262144)}
video_max_pixels: 16384
trust_remote_code: true

### method
{chr(10).join(method_lines)}

### dataset
dataset_dir: {dataset_dir}
dataset: {dataset_name}
eval_dataset: dota128_ref_val_official
template: {template}
cutoff_len: {training['cutoff_len']}
max_samples: 999999999999
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: {output_dir}
logging_steps: 10
save_strategy: epoch
eval_strategy: epoch
plot_loss: true
overwrite_output_dir: false
save_only_model: false
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
ddp_timeout: 180000000
resume_from_checkpoint: null
optim: adamw_torch
weight_decay: 0.01
adam_beta1: 0.9
adam_beta2: 0.95
seed: {settings['project']['seed']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=None)
    args = parser.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    training = settings["training"]
    config_dir = Path(settings["paths"]["training_run_root"]) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    direct_cfg = config_dir / "b1_direct_official.yaml"
    socratic_cfg = config_dir / "b2_socratic_official.yaml"
    direct_cfg.write_text(
        render(settings, "dota128_b1_matched_train_official", training["b1_adapter_output"]),
        encoding="utf-8",
    )
    socratic_cfg.write_text(
        render(settings, "dota128_b2_matched_train_official", training["b2_adapter_output"]),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "direct_config": str(direct_cfg),
                "socratic_config": str(socratic_cfg),
                "base_model_key": training["base_model_key"],
                "template_resolved": resolve_training_template(settings),
                "meaning": {
                    "B1": "Direct OBB plus final-only Ref pairs",
                    "B2": "The same Direct rows plus Socratic versions of exactly the same Ref pairs",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
