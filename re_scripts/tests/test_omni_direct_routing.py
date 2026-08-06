from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_omni_materialization_preserves_rs_eot(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    original_rs = json.loads(json.dumps(base["models"]["rs_eot"]))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    classes = dataset / "classes.txt"
    classes.write_text("vehicle\\n", encoding="utf-8")
    output_root = tmp_path / "out"
    output = output_root / "config" / "settings.zh.v1.2.3.json"
    cmd = [
        sys.executable,
        str(root / "tools/materialize_language_settings.py"),
        "--base-settings", str(root / "settings.json"),
        "--classes-file", str(classes),
        "--lang", "zh",
        "--output-root", str(output_root),
        "--output", str(output),
        "--dataset-root", str(dataset),
        "--dataset-variant", "raw",
        "--eval-gpus", "1,2,3",
        "--eval-topology", "replica",
        "--sft-visible-gpus", "1,2,3",
        "--sft-world-size", "3",
        "--sft-gradient-accum", "3",
        "--student-model-family", "qwen3_omni",
        "--student-base-model-path", "/models/qwen3-omni",
        "--student-template", "qwen3_omni",
        "--student-b1-adapter-output", "/adapters/b1",
        "--student-b2-adapter-output", "/adapters/b2",
        "--student-adapter-only", "1",
        "--student-freeze-vision", "1",
        "--student-freeze-projector", "1",
        "--student-lora-target", "all",
    ]
    subprocess.run(cmd, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["models"]["rs_eot"] == original_rs
    assert result["model_routing"]["b0_model_key"] == "qwen3_omni_b0"
    assert result["model_routing"]["b1_model_key"] == "qwen3_omni_b1_direct"
    assert result["models"]["qwen3_omni_b1_direct"]["adapter_path"] == "/adapters/b1"
    assert result["training"]["base_model_key"] == "qwen3_omni_b0"
    assert result["training"]["adapter_only"] is True
    assert result["training"]["freeze_vision_tower"] is True
    assert result["training"]["freeze_multi_modal_projector"] is True
    assert result["v1_2_3"]["base_settings_sha256"]
