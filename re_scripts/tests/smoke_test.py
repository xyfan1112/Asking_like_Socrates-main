#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="re_scripts_smoke_") as td:
    t = Path(td)
    d = t / "datasets/dota128"
    cases = [
        (
            "train",
            "a",
            [
                "1 .1 .1 .2 .1 .2 .2 .1 .2",
                "10 .6 .6 .7 .6 .7 .7 .6 .7",
                "1 -.1 .2 .05 .2 .05 .3 -.1 .3",  # invalid edge box, must be dropped
            ],
        ),
        ("val", "b", ["1 .2 .7 .3 .7 .3 .8 .2 .8"]),
    ]
    for split, name, lines in cases:
        (d / split / "images").mkdir(parents=True)
        (d / split / "labels").mkdir(parents=True)
        Image.new("RGB", (512, 512), "white").save(d / split / "images" / f"{name}.jpg")
        (d / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")

    s = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
    p = s["paths"]
    p["dota128_root"] = str(d)
    p["dota128_ref_root"] = str(t / "datasets/dota128-Ref")
    p["dota128_llamafactory_root"] = str(t / "datasets/dota128-llamafactory")
    p["dota128_rl_root"] = str(t / "datasets/dota128-rl")
    p["pipeline_work_root"] = str(t / "results/pipeline")
    p["trajectory_raw_dir"] = str(t / "results/pipeline/trajectory_raw")
    p["trajectory_audit_dir"] = str(t / "results/pipeline/trajectory_audit")
    p["training_run_root"] = str(t / "results/training")
    p["test_run_root"] = str(t / "results/testing")
    s["data_conversion"]["semantic_mode"] = "deterministic"
    s["data_conversion"]["use_scene_description"] = False
    s["data_quality"]["invalid_object_policy"] = "drop"
    sp = t / "settings.json"
    sp.write_text(json.dumps(s), encoding="utf-8")

    scripts = [
        "data_layer/01_audit_dota128.py",
        "data_layer/02_build_dota128_ref.py",
        "data_layer/03_validate_dota128_ref.py",
        "data_layer/04_visualize_dota128_ref.py",
        "data_layer/05_build_agent_inputs.py",
        "data_layer/05c_audit_lineage.py",
        "data_layer/05b_verify_data_outputs.py",
    ]
    for script in scripts:
        subprocess.run(
            [sys.executable, str(ROOT / script), "--settings", str(sp)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    build = json.loads((Path(p["dota128_ref_root"]) / "build_report.json").read_text())
    train = build["splits"]["train"]
    assert train["objects_raw"] == 3
    assert train["objects_valid"] == 2
    assert train["objects_dropped_invalid"] == 1
    validation = json.loads((Path(p["dota128_ref_root"]) / "validation_report.json").read_text())
    assert validation["passed"] is True
    gate = json.loads((Path(p["pipeline_work_root"]) / "reports/data_layer_final_report.json").read_text())
    assert gate["passed"] is True
    print("SMOKE TEST PASS")
