#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_data_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="re_scripts_v4_contract_") as td:
        temp = Path(td)
        dota = temp / "datasets/dota128"
        for split, name in (("train", "train_a"), ("val", "val_b")):
            (dota / split / "images").mkdir(parents=True)
            (dota / split / "labels").mkdir(parents=True)
            Image.new("RGB", (400, 300), "white").save(dota / split / "images" / f"{name}.jpg")
            # two ships and one small vehicle; normalized DOTA-style OBB rows
            lines = [
                "1 .05 .2 .15 .2 .15 .7 .05 .7",
                "1 .30 .2 .40 .2 .40 .7 .30 .7",
                "10 .70 .1 .80 .1 .80 .25 .70 .25",
            ]
            (dota / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
        paths = settings["paths"]
        paths.update({
            "dota128_root": str(dota),
            "dota128_ref_root": str(temp / "datasets/dota128-Ref"),
            "dota128_llamafactory_root": str(temp / "datasets/dota128-llamafactory"),
            "dota128_rl_root": str(temp / "datasets/dota128-rl"),
            "pipeline_work_root": str(temp / "results/pipeline"),
            "trajectory_raw_dir": str(temp / "results/pipeline/raw"),
            "trajectory_audit_dir": str(temp / "results/pipeline/audit"),
            "training_run_root": str(temp / "results/training"),
            "rl_run_root": str(temp / "results/rl"),
            "test_run_root": str(temp / "results/testing"),
        })
        settings_path = temp / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        for rel in (
            "data_layer/01_audit_dota128.py",
            "data_layer/02_build_dota128_ref.py",
            "data_layer/03_validate_dota128_ref.py",
            "data_layer/05_build_agent_inputs.py",
            "train_layer/rl/01_build_grounding_rl_data.py",
        ):
            subprocess.run(
                [sys.executable, str(ROOT / rel), "--settings", str(settings_path)],
                check=True,
                stdout=subprocess.DEVNULL,
            )

        selected = [json.loads(x) for x in (temp / "datasets/dota128-Ref/train.jsonl").read_text(encoding="utf-8").splitlines()]
        all_rows = [json.loads(x) for x in (temp / "datasets/dota128-Ref/train_all.jsonl").read_text(encoding="utf-8").splitlines()]
        assert selected and len(all_rows) == 3
        for row in selected:
            assert row["schema_version"] == "dota_ref_v4_3_3"
            assert row["reference_grounding"]
            assert row["reference_classification"]
            assert row["class_name"] not in row["reference_classification"].lower()
            assert not any(token in row["reference_grounding"].lower() for token in ("1st", "2nd", "3rd", "first", "second", "third"))
            assert "of its class" not in row["reference_grounding"].lower()
            assert "of its class" not in row["reference_classification"].lower()
            assert len(row["obb_pixel"]) == len(row["obb_norm100"]) == len(row["obb_norm1000"]) == 4
            assert len(row["hbb_norm1000"]) == 4
            assert len(row["classification_focus_hbb_norm1000"]) == 4
            assert "coarse focus region" in row["grounding_question"].lower()
            expected_focus = "[" + ",".join(
                str(int(round(float(value))))
                for value in row["classification_focus_hbb_norm1000"]
            ) + "]"
            assert expected_focus in row["grounding_question"].replace(" ", "")
            assert expected_focus in row["classification_question"].replace(" ", "")
            assert "area area" not in row["reference_grounding"].lower()
            assert row["reference_semantics"]["semantic_unique"] is True
            assert row["reference_semantics"]["match_object_indices"] == [
                row["object_index"]
            ]
            if row["class_name"] in {"large vehicle", "small vehicle"}:
                assert "small large vehicle" not in row["reference_grounding"].lower()
                assert "large large vehicle" not in row["reference_grounding"].lower()

        agent_rows = [json.loads(x) for x in (temp / "results/pipeline/agent_inputs/train_agent_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {r["task"] for r in agent_rows} == {"ref_grounding_obb", "ref_classification"}
        assert any(r["reference_type"] == "classification_masked_label" for r in agent_rows)
        assert any("normalized coordinates in [0,1000]" in r["query"] for r in agent_rows if r["task"] == "ref_grounding_obb")
        assert all(
            "focus region" in r["query"].lower()
            for r in agent_rows
            if r["task"] in {"ref_grounding_obb", "ref_classification"}
        )

        rl_rows = [json.loads(x) for x in (temp / "datasets/dota128-rl/train.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rl_rows
        assert all(len(r["answer"]["bbox"]) == 4 for r in rl_rows)
        assert all(0 <= v <= 1000 for r in rl_rows for v in r["answer"]["bbox"])


def run_reward_contract() -> None:
    reward = load_module("dota_hbb_reward", ROOT / "train_layer/rl/rewards/dota_hbb_reward.py")
    response = '<think>check target</think>\nAnswer: {"class_name":"truck","bbox":[100,200,300,400]}'
    result = reward.compute_score([{
        "response": response,
        "ground_truth": {"class_name": "large vehicle", "bbox": [100, 200, 300, 400]},
    }])[0]
    assert result["format"] == 1.0
    assert result["class"] == 1.0
    assert abs(result["IOU"] - 1.0) < 1e-9
    assert abs(result["overall"] - 1.0) < 1e-9


def run_teacher_force_contract() -> None:
    converter = load_module(
        "official_converter",
        ROOT / "data_layer/official_socratic/04_convert_official_to_llamafactory.py",
    )
    trace = "inspect region\n</think>wrong|1,2,3,4,5,6,7,8"
    forced, generated = converter.replace_final_with_gt(trace, "ship|10,20,30,40,50,60,70,80", True)
    assert generated == "wrong|1,2,3,4,5,6,7,8"
    assert forced.endswith("</think>ship|10,20,30,40,50,60,70,80")
    normalized = converter.normalize_assistant_content("<think>" + forced, "qwen2_vl_thinking")
    assert not normalized.lstrip().startswith("<think>")
    assert "</think>" in normalized


def run_strict_parser_contract() -> None:
    sys.path.insert(0, str(ROOT / "main_layer"))
    from common import parse_obb_output_status

    malformed = "<think>1 2 3 4 5 6 7 8</think>\nThe box may be ship 1 2 3 4 5 6 7 8"
    bad = parse_obb_output_status(
        malformed,
        1000,
        1000,
        "normalized_0_1000",
        strict=True,
        require_markers=True,
        expected_class="ship",
        require_confidence=True,
    )
    assert bad["predictions"] == []
    assert bad["format_ok"] is False
    valid = parse_obb_output_status(
        "FINAL_DETECTIONS\nship|0.9|10,10,20,10,20,20,10,20\nEND_DETECTIONS",
        1000,
        1000,
        "normalized_0_1000",
        strict=True,
        require_markers=True,
        expected_class="ship",
        require_confidence=True,
    )
    assert valid["format_ok"] is True
    assert len(valid["predictions"]) == 1



def run_profile_contract() -> None:
    config = load_module("runtime_config", ROOT / "main_layer/config.py")
    with tempfile.TemporaryDirectory(prefix="re_scripts_profile_") as td:
        settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
        settings["paths"]["results_root"] = str(Path(td) / "results")
        settings_path = Path(td) / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        out = config.materialize_settings(
            settings_path, str(ROOT / "experiments/a3_no_dota_alias.json")
        )
        merged = json.loads(Path(out).read_text(encoding="utf-8"))
        assert merged["trajectory"]["prompt_profile"] == "obb_grounding_no_alias"
        assert merged["trajectory"]["geometry_gate"]["classification_alias_mapping"] is False


def main() -> None:
    run_data_contract()
    run_reward_contract()
    run_teacher_force_contract()
    run_strict_parser_contract()
    run_profile_contract()
    print("V4.3.3 CONTRACT TESTS PASS")


if __name__ == "__main__":
    main()
