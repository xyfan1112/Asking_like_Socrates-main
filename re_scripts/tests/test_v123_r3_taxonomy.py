#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))

from taxonomy import (  # noqa: E402
    materialize_runtime_taxonomy,
    resolve_classes_file_path,
)


def test_polluted_shell_value_recovers_only_existing_file(tmp_path: Path) -> None:
    classes = tmp_path / "classes.zh.txt"
    classes.write_text("雷达车\n发射车\n", encoding="utf-8")
    polluted = (
        "[PASS] scene-disjoint manifest: train=128 val=20 overlap=0\n"
        + str(classes)
    )
    assert resolve_classes_file_path(polluted) == classes.resolve()


def test_runtime_taxonomy_rewrites_polluted_cli_value_to_clean_path(tmp_path: Path) -> None:
    classes = tmp_path / "classes.zh.txt"
    classes.write_text("雷达车\n发射车\n", encoding="utf-8")
    polluted = "[INFO] stale log line\n" + str(classes)
    settings = {"taxonomy": {"classes_file": str(classes), "qa_language": "zh"}}
    catalog = materialize_runtime_taxonomy(
        settings, classes_file=polluted, language="zh"
    )
    assert catalog.classes_file == str(classes.resolve())
    assert settings["taxonomy"]["classes_file"] == str(classes.resolve())
    assert settings["taxonomy"]["num_classes"] == 2


def test_active_preflight_rejects_polluted_settings_value(tmp_path: Path) -> None:
    classes = tmp_path / "classes.zh.txt"
    classes.write_text("雷达车\n发射车\n", encoding="utf-8")
    results = tmp_path / "results"
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "paths": {"results_root": str(results)},
                "taxonomy": {
                    "classes_file": "[PASS] log\n" + str(classes),
                    "source_classes_file": str(classes),
                    "qa_language": "zh",
                },
                "v1_2_3": {"class_count": 2},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/validate_active_taxonomy.py"),
            "--settings",
            str(settings),
            "--expected-lang",
            "zh",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.returncode != 0
    assert "污染" in proc.stdout


def test_r3_runner_has_no_shell_path_command_substitution() -> None:
    runner = (ROOT / "commands/run_v1.2.3.sh").read_text(encoding="utf-8")
    assert 'RELEASE_REVISION="1.2.3-r3"' in runner
    assert "classes=$(active_classes)" not in runner
    assert "active_classes() {" not in runner
    assert '--classes-file "$classes"' not in runner
    assert "validate_active_taxonomy.py" in runner
