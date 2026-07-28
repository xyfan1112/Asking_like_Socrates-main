#!/usr/bin/env python3
"""Create a scene-disjoint DOTA128 copy without modifying the source dataset."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_layer"))
from common import discover_images, load_settings, settings_from_cli, write_json  # noqa: E402


def scene_id(path: Path) -> str:
    return path.stem.split("__", 1)[0]


def materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--output-root")
    ap.add_argument(
        "--settings-output",
        help="Write a settings copy whose paths.dota128_root points to the new dataset.",
    )
    args = ap.parse_args()
    settings_path = settings_from_cli(__file__, args.settings)
    settings = load_settings(settings_path)
    source_root = Path(settings["paths"]["dota128_root"]).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else Path(settings["paths"]["datasets_root"]).expanduser().resolve()
        / "dota128_scene_disjoint"
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"Output is not empty: {output_root}. Choose a new directory; the tool will not delete data."
        )

    records: list[dict] = []
    original_counts: Counter[str] = Counter()
    original_scene_splits: dict[str, Counter[str]] = defaultdict(Counter)
    for split in ("train", "val"):
        for image, label in discover_images(source_root, split):
            scene = scene_id(image)
            records.append(
                {
                    "scene_id": scene,
                    "original_split": split,
                    "image": image,
                    "label": label,
                }
            )
            original_counts[split] += 1
            original_scene_splits[scene][split] += 1
    if not records:
        raise RuntimeError(f"No DOTA images found under {source_root}")

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scene[record["scene_id"]].append(record)

    # Start from scenes that were exclusively validation. Leaked scenes go to
    # the split containing most of their tiles (train wins deterministic ties).
    assignment: dict[str, str] = {}
    for scene, counts in original_scene_splits.items():
        if counts["val"] > 0 and counts["train"] == 0:
            assignment[scene] = "val"
        elif counts["val"] > counts["train"]:
            assignment[scene] = "val"
        else:
            assignment[scene] = "train"

    target_val_images = original_counts["val"]
    rng = random.Random(int(settings["project"].get("seed", 42)))

    def val_count() -> int:
        return sum(
            len(by_scene[scene]) for scene, split in assignment.items() if split == "val"
        )

    if val_count() < target_val_images:
        candidates = [scene for scene, split in assignment.items() if split == "train"]
        rng.shuffle(candidates)
        candidates.sort(key=lambda scene: (len(by_scene[scene]), scene))
        while candidates and val_count() < target_val_images:
            assignment[candidates.pop(0)] = "val"
    elif val_count() > target_val_images:
        candidates = [scene for scene, split in assignment.items() if split == "val"]
        rng.shuffle(candidates)
        candidates.sort(key=lambda scene: (-len(by_scene[scene]), scene))
        while len(candidates) > 1 and val_count() > target_val_images:
            scene = candidates.pop(0)
            current = val_count()
            if abs((current - len(by_scene[scene])) - target_val_images) <= abs(
                current - target_val_images
            ):
                assignment[scene] = "train"

    link_modes: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    manifest_rows = []
    for record in sorted(records, key=lambda item: str(item["image"])):
        split = assignment[record["scene_id"]]
        image_dst = output_root / split / "images" / record["image"].name
        label_dst = output_root / split / "labels" / record["label"].name
        link_modes[materialize(record["image"], image_dst)] += 1
        link_modes[materialize(record["label"], label_dst)] += 1
        output_counts[split] += 1
        manifest_rows.append(
            {
                "scene_id": record["scene_id"],
                "original_split": record["original_split"],
                "new_split": split,
                "image": record["image"].name,
                "label": record["label"].name,
            }
        )

    train_scenes = {scene for scene, split in assignment.items() if split == "train"}
    val_scenes = {scene for scene, split in assignment.items() if split == "val"}
    overlap = sorted(train_scenes & val_scenes)
    report = {
        "schema_version": "scene_disjoint_split_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "original_image_counts": dict(original_counts),
        "output_image_counts": dict(output_counts),
        "target_val_images": target_val_images,
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "scene_overlap": overlap,
        "materialization_modes": dict(link_modes),
        "rows": manifest_rows,
        "passed": not overlap and output_counts["train"] > 0 and output_counts["val"] > 0,
    }
    write_json(output_root / "scene_split_manifest.json", report)

    settings_output = (
        Path(args.settings_output).expanduser().resolve()
        if args.settings_output
        else settings_path.with_name("settings.scene_disjoint.json")
    )
    new_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    new_settings["paths"]["dota128_root"] = str(output_root)
    write_json(settings_output, new_settings)
    print(json.dumps({**report, "rows": f"{len(manifest_rows)} rows"}, ensure_ascii=False, indent=2))
    print(f"[SETTINGS] {settings_output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
