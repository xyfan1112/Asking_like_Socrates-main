#!/usr/bin/env python3
"""Prevent incompatible official Socratic runs from being resumed together."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import load_settings, settings_from_cli, write_json  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(settings: dict[str, Any], split: str) -> dict[str, Any]:
    official = settings["official_socratic"]
    parquet = Path(official["input_parquet_dir"]) / f"dota128_{split}_official.parquet"
    files = {
        "parquet": parquet,
        "adapter": ROOT / "data_layer/official_socratic/local_api_adapter/utils.py",
        "prompts": ROOT / "data_layer/prompts/profiles.py",
        "ref_builder": ROOT / "data_layer/02_build_dota128_ref.py",
        "ref_semantics": ROOT / "data_layer/ref_semantics.py",
        "trajectory_gates": ROOT / "data_layer/trajectory_gates.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Signature inputs missing: {missing}")
    version = json.loads((ROOT / "VERSION.json").read_text(encoding="utf-8"))
    payload = {
        "package_version": version.get("version"),
        "split": split,
        "prompt_profile": settings["trajectory"].get("prompt_profile"),
        "coordinate_target": settings["data_conversion"].get("coordinate_target"),
        "max_loop": settings["official_socratic"].get("max_loop"),
        "force_final": settings["trajectory"].get("force_final_on_last_round"),
        "repair_attempts": settings["trajectory"].get("max_repair_attempts"),
        "question_similarity_threshold": settings["trajectory"].get(
            "question_similarity_threshold"
        ),
        "require_perceiver_original_context": settings["trajectory"].get(
            "require_perceiver_original_context"
        ),
        "geometry_gate": settings["trajectory"].get("geometry_gate"),
        "visibility": {
            "min_short_side_norm1000": settings["data_conversion"].get(
                "socratic_min_short_side_norm1000"
            ),
            "min_area_ratio": settings["data_conversion"].get(
                "socratic_min_area_ratio"
            ),
        },
        "hashes": {name: sha256(path) for name, path in files.items()},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["signature"] = hashlib.sha256(encoded).hexdigest()
    return payload


def related_paths(settings: dict[str, Any], split: str) -> list[Path]:
    official = settings["official_socratic"]
    raw = Path(official["raw_output_dir"])
    post = Path(official["postproc_output_dir"])
    lf = Path(settings["paths"]["dota128_llamafactory_root"])
    return [
        raw / f"dota128_{split}_official.jsonl",
        raw / f"dota128_{split}_official_strict.jsonl",
        raw / f"dota128_{split}_official_strict.audit.json",
        raw / f"dota128_{split}_official_strict.report.json",
        post / f"dota128_{split}_official_merge.json",
        lf / f"dota128_socratic_{split}_official.json",
        lf / "dota128_mixed_train_official.json" if split == "train" else lf / "__unused__",
        lf / "dota128_b1_matched_train_official.json" if split == "train" else lf / "__unused_b1__",
        lf / "dota128_b2_matched_train_official.json" if split == "train" else lf / "__unused_b2__",
        lf / "dota128_ref_val_official.json" if split == "train" else lf / "__unused_val__",
    ]


def archive_existing(settings: dict[str, Any], split: str) -> Path | None:
    official = settings["official_socratic"]
    root = Path(official["raw_output_dir"]).parent / "archive"
    existing = [path for path in related_paths(settings, split) if path.exists()]
    manifest = Path(official["raw_output_dir"]) / f"dota128_{split}_official.manifest.json"
    if manifest.exists():
        existing.append(manifest)
    if not existing:
        return None
    archive = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        target = archive / path.name
        shutil.move(str(path), str(target))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "finalize", "reset", "show"])
    parser.add_argument("--settings")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--mode", default="full", choices=["full", "debug"])
    parser.add_argument("--policy", default="auto", choices=["auto", "fresh", "resume"])
    args = parser.parse_args()

    settings = load_settings(settings_from_cli(__file__, args.settings))
    official = settings["official_socratic"]
    raw_dir = Path(official["raw_output_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical = raw_dir / f"dota128_{args.split}_official.jsonl"
    manifest_path = raw_dir / f"dota128_{args.split}_official.manifest.json"
    sig = signature(settings, args.split)

    if args.action == "reset":
        archive = archive_existing(settings, args.split)
        print(json.dumps({"archived_to": str(archive) if archive else None}, ensure_ascii=False))
        return 0
    if args.action == "show":
        print(json.dumps(sig, ensure_ascii=False, indent=2))
        return 0

    if args.action == "prepare":
        # Debug runs must never overwrite the canonical full-run manifest.  In
        # v4.2.x a debug launch could refresh the manifest beside an old raw
        # file, making a later full --resume incorrectly accept mixed prompts.
        if args.mode == "debug":
            debug_manifest = raw_dir / f"dota128_{args.split}_official_debug.manifest.json"
            pending = {
                **sig,
                "status": "prepared",
                "mode": "debug",
                "policy": "fresh",
                "resume": False,
                "archive": None,
            }
            write_json(debug_manifest, pending)
            print(json.dumps({"resume": False, "manifest": str(debug_manifest), "archive": None}, ensure_ascii=False))
            return 0

        resume = False
        archive = None
        if args.policy == "fresh":
            archive = archive_existing(settings, args.split)
            resume = False
        elif canonical.exists():
            if not manifest_path.exists():
                raise SystemExit(
                    "[FAIL] Existing canonical raw has no v4.3 manifest. It may mix old prompts. "
                    "Run full with policy=fresh."
                )
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("signature") != sig["signature"]:
                raise SystemExit(
                    "[FAIL] Existing raw signature differs from current query/prompt/code. "
                    "Do not resume. Run full with policy=fresh."
                )
            resume = True
        elif args.policy == "resume":
            raise SystemExit("[FAIL] resume requested but canonical raw does not exist")

        pending = {
            **sig,
            "status": "prepared",
            "mode": args.mode,
            "policy": args.policy,
            "resume": resume,
            "archive": str(archive) if archive else None,
        }
        write_json(manifest_path, pending)
        print(json.dumps({"resume": resume, "manifest": str(manifest_path), "archive": pending["archive"]}, ensure_ascii=False))
        return 0

    if not canonical.exists():
        raise SystemExit(f"[FAIL] canonical raw missing after generation: {canonical}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else sig
    rows = sum(1 for line in canonical.open(encoding="utf-8") if line.strip())
    manifest.update({"status": "completed", "rows": rows})
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
