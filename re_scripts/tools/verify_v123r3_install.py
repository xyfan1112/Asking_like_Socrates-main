#!/usr/bin/env python3
"""Verify that the files actually executed are the v1.2.3-r3 files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REVISION = "1.2.3-r3"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    required = {
        "commands/run_v1.2.3.sh": [
            'RELEASE_REVISION="1.2.3-r3"',
            "validate_active_taxonomy.py",
            "main_layer/run.py data-full --settings \"$ACTIVE_CFG\" --lang \"$RUN_LANG\"",
        ],
        "main_layer/run.py": ['RELEASE_REVISION = "1.2.3-r3"'],
        "main_layer/taxonomy.py": [
            "def resolve_classes_file_path(",
            "[TAXONOMY RECOVER][v1.2.3-r3]",
        ],
        "tools/validate_active_taxonomy.py": ['REVISION = "1.2.3-r3"'],
    }
    errors: list[str] = []
    for rel, markers in required.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(f"marker missing in {rel}: {marker!r}")

    runner = root / "commands/run_v1.2.3.sh"
    if runner.is_file():
        text = runner.read_text(encoding="utf-8")
        forbidden = {
            "classes=$(active_classes)": "旧命令替换仍存在",
            '--classes-file "$classes"': "下游仍显式传递Shell解析路径",
            "active_classes() {": "旧active_classes函数仍存在",
        }
        for token, reason in forbidden.items():
            if token in text:
                errors.append(f"{reason}: {token}")

    if errors:
        print(f"[VERIFY INSTALL][{REVISION}] FAIL")
        for item in errors:
            print(f"  - {item}")
        return 2

    print(f"[VERIFY INSTALL][{REVISION}] PASS")
    print(f"  root = {root}")
    for rel in required:
        path = root / rel
        print(f"  {rel}: {sha256(path)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
