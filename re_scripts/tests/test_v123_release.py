#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def test_bilingual_materializer_preserves_effective_batch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    scene = tmp_path / "scene"
    raw.mkdir(); scene.mkdir()
    classes = raw / "classes.txt"
    classes.write_text("发射车\nRadar vehicle\n", encoding="utf-8")
    output_root = tmp_path / "out"
    output = output_root / "config/settings.en.v1.2.3.json"
    class_map = output_root / "config/class_map.zh_en.json"
    cmd = [
        str(ROOT / "tools/materialize_language_settings.py"),
        "--base-settings", str(ROOT / "settings.json"),
        "--classes-file", str(classes),
        "--lang", "en",
        "--output-root", str(output_root),
        "--output", str(output),
        "--dataset-root", str(scene),
        "--dataset-variant", "scene_disjoint",
        "--class-map", str(class_map),
        "--sft-world-size", "4",
        "--sft-gradient-accum", "auto",
        "--require-frozen-vision",
    ]
    first = run(*cmd, check=False)
    assert first.returncode == 3
    mapping = json.loads(class_map.read_text(encoding="utf-8"))
    mapping["classes"][0]["en"] = "Missile launcher vehicle"
    class_map.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    run(*cmd)
    settings = json.loads(output.read_text(encoding="utf-8"))
    assert settings["v1_2_3"]["language"] == "en"
    assert settings["training"]["world_size"] == 4
    assert settings["training"]["gradient_accumulation_steps"] == 2
    assert settings["v1_2_3"]["sft"]["effective_global_batch"] == 8
    assert settings["trajectory"]["prompt_profile"] == "custom_obb_bilingual_v1_2_3"


def test_replica_shard_merger_is_deterministic(tmp_path: Path) -> None:
    shards = []
    rows = [("b", 0), ("a", 1), ("a", 0), ("c", 0)]
    for idx, (ident, run_id) in enumerate(rows):
        shard = tmp_path / f"s{idx}.jsonl"
        shard.write_text(json.dumps({"id": ident, "run_id": run_id}) + "\n", encoding="utf-8")
        shard.with_suffix(".manifest.json").write_text('{"expected_rows":1}\n', encoding="utf-8")
        shards.append(shard)
    output = tmp_path / "all.jsonl"
    run(
        str(ROOT / "tools/merge_jsonl_shards.py"),
        "--output", str(output),
        "--shards", *map(str, shards),
    )
    merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(x["id"], x["run_id"]) for x in merged] == [
        ("a", 0), ("a", 1), ("b", 0), ("c", 0)
    ]


def test_b2_must_contain_exact_direct_prefix(tmp_path: Path) -> None:
    lf = tmp_path / "lf"; lf.mkdir()
    training = tmp_path / "training"
    pipeline = tmp_path / "pipeline"
    direct = [{"id": 1}, {"id": 2}]
    (lf / "dota128_direct_train_official.json").write_text(json.dumps(direct), encoding="utf-8")
    (lf / "dota128_b2_matched_train_official.json").write_text(
        json.dumps(direct + [{"id": "socratic"}]), encoding="utf-8"
    )
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"paths": {
        "dota128_llamafactory_root": str(lf),
        "training_run_root": str(training),
        "pipeline_work_root": str(pipeline),
    }}), encoding="utf-8")
    result = run(str(ROOT / "tools/audit_b1_b2_direct_pairing.py"), "--settings", str(settings))
    assert "PASS" in result.stdout
    report = json.loads((training / "reports/b1_b2_direct_pairing_report.json").read_text(encoding="utf-8"))
    assert report["direct_prefix_exact"] is True
    assert report["b2_socratic_rows"] == 1


def test_release_prompt_alias_exists() -> None:
    script = (
        "import sys; "
        f"sys.path.insert(0,{str(ROOT)!r}); "
        "from data_layer.prompts.profiles import PROFILES; "
        "assert 'custom_obb_bilingual_v1_2_3' in PROFILES"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
