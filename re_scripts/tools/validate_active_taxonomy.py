#!/usr/bin/env python3
"""Strict v1.2.3-r3 active-taxonomy preflight.

This checker validates the path stored in the generated language settings.  It
never accepts shell progress text as part of a path, and it records a small JSON
report under the run's results/reports directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from taxonomy import load_class_names, resolve_classes_file_path  # noqa: E402

REVISION = "1.2.3-r3"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_report(settings: dict[str, Any], payload: dict[str, Any]) -> Path | None:
    results_root = settings.get("paths", {}).get("results_root")
    if not results_root:
        return None
    path = Path(str(results_root)).expanduser().resolve() / "reports" / "taxonomy_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--expected-lang", choices=("zh", "en"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    settings_path = args.settings.expanduser().resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"active settings不存在: {settings_path}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    taxonomy = settings.get("taxonomy")
    if not isinstance(taxonomy, dict):
        raise ValueError("active settings缺少taxonomy对象")

    raw = taxonomy.get("classes_file")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"taxonomy.classes_file为空或不是字符串: {raw!r}")
    if any(ch in raw for ch in ("\n", "\r", "\x00")):
        raise ValueError(
            "taxonomy.classes_file已被日志或控制字符污染；请重新运行r3 prepare。"
            f" raw={raw!r}"
        )

    active_path = resolve_classes_file_path(raw, origin="active settings taxonomy.classes_file")
    active_labels = load_class_names(active_path)
    language = str(taxonomy.get("qa_language") or "").strip().lower()
    if language not in {"zh", "en"}:
        raise ValueError(f"taxonomy.qa_language非法: {language!r}")
    if args.expected_lang and language != args.expected_lang:
        raise ValueError(
            f"语言不一致: settings={language}, expected={args.expected_lang}"
        )

    source_raw = taxonomy.get("source_classes_file")
    source_path = None
    source_labels = None
    if source_raw:
        source_path = resolve_classes_file_path(
            str(source_raw), origin="active settings taxonomy.source_classes_file"
        )
        source_labels = load_class_names(source_path)
        if len(source_labels) != len(active_labels):
            raise ValueError(
                f"源类别与活动类别数量不同: source={len(source_labels)} active={len(active_labels)}"
            )
        if language == "zh" and tuple(source_labels) != tuple(active_labels):
            raise ValueError("中文活动类别与源classes.txt顺序/内容不一致")

    expected_count = settings.get("v1_2_3", {}).get("class_count")
    if expected_count is not None and int(expected_count) != len(active_labels):
        raise ValueError(
            f"v1_2_3.class_count不一致: signature={expected_count} actual={len(active_labels)}"
        )

    canonical_payload = json.dumps(
        list(active_labels), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    payload: dict[str, Any] = {
        "schema_version": "taxonomy_preflight_v1_2_3_r3",
        "release_revision": REVISION,
        "passed": True,
        "settings": str(settings_path),
        "language": language,
        "active_classes_file": str(active_path),
        "active_classes_count": len(active_labels),
        "active_taxonomy_sha256": sha256_bytes(canonical_payload),
        "source_classes_file": str(source_path) if source_path else None,
        "source_classes_count": len(source_labels) if source_labels is not None else None,
        "first_labels": list(active_labels[:5]),
    }
    report = _write_report(settings, payload)
    if not args.quiet:
        print(f"[TAXONOMY PREFLIGHT][{REVISION}] PASS")
        print(f"  settings = {settings_path}")
        print(f"  language = {language}")
        print(f"  classes  = {active_path}")
        print(f"  count    = {len(active_labels)}")
        if report:
            print(f"  report   = {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
