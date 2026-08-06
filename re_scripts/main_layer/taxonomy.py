"""Runtime-selectable class taxonomy and QA language.

The stable 2A6000 pipeline historically imported ``DOTA_CLASSES`` at module
load time.  This module keeps that contract while allowing ``main_layer/run.py``
to provide an arbitrary UTF-8 ``classes.txt`` through environment variables.

No example classes are embedded in custom mode.  The line order in the supplied
file is the class-id order: first usable line -> id 0, second -> id 1, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from typing import Iterable

DEFAULT_DOTA_CLASSES = (
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
)

DEFAULT_DOTA_ALIASES = {
    "storage-tank": "storage tank", "baseball-diamond": "baseball diamond",
    "tennis-court": "tennis court", "basketball-court": "basketball court",
    "ground-track-field": "ground track field", "large-vehicle": "large vehicle",
    "small-vehicle": "small vehicle", "soccer-ball-field": "soccer ball field",
    "swimming-pool": "swimming pool",
    "bus": "large vehicle", "truck": "large vehicle", "lorry": "large vehicle",
    "trailer": "large vehicle", "truck trailer": "large vehicle",
    "tractor trailer": "large vehicle", "heavy vehicle": "large vehicle",
    "car": "small vehicle", "sedan": "small vehicle", "suv": "small vehicle",
    "pickup": "small vehicle", "pickup truck": "small vehicle",
    "compact van": "small vehicle", "airplane": "plane",
    "aircraft": "plane", "jet": "plane", "boat": "ship", "vessel": "ship",
}

_FORBIDDEN_LABEL_PARTS = (
    "|",
    "\n",
    "\r",
    "FINAL_DETECTIONS",
    "END_DETECTIONS",
)


def normalize_lookup_key(value: object) -> str:
    """Normalize only for matching, never for final-answer spelling."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;[](){}\"'")


def _usable_lines(text: str) -> list[str]:
    rows: list[str] = []
    for raw in text.splitlines():
        line = unicodedata.normalize("NFC", raw).strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


def resolve_classes_file_path(
    classes_file: str | Path,
    *,
    must_exist: bool = True,
    origin: str = "classes_file",
) -> Path:
    """Resolve a classes-file value without allowing shell-log contamination.

    v1.2.3-r1 could accidentally capture a progress line together with the real
    path through Bash command substitution.  The resulting value looked like::

        [PASS] scene-disjoint manifest: ...\n/home/.../classes.zh.txt

    Normal paths must never contain a newline.  For backward recovery, when the
    value has multiple non-empty lines and exactly one line is an existing file,
    that file is selected and a loud recovery warning is emitted.  Ambiguous or
    non-existent values fail before any data conversion starts.
    """
    raw = str(classes_file or "")
    if "\x00" in raw:
        raise ValueError(f"{origin} contains NUL byte: {raw!r}")
    lines = [line.strip() for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        raise ValueError(f"{origin} is empty")

    def _candidate(line: str) -> Path:
        return Path(line).expanduser().resolve()

    if len(lines) > 1:
        existing = []
        for line in lines:
            try:
                candidate = _candidate(line)
            except (OSError, ValueError):
                continue
            if candidate.is_file():
                existing.append(candidate)
        unique = list(dict.fromkeys(existing))
        if len(unique) == 1:
            recovered = unique[0]
            print(
                f"[TAXONOMY RECOVER][v1.2.3-r3] {origin} contained "
                f"{len(lines)} lines; using the only existing file: {recovered}",
                file=sys.stderr,
            )
            return recovered
        raise ValueError(
            f"{origin} contains multiple lines and cannot be resolved safely: "
            f"raw={raw!r}, existing_candidates={[str(x) for x in unique]!r}"
        )

    path = _candidate(lines[0])
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"classes.txt 不存在: {path}")
    return path


def load_class_names(classes_file: str | Path) -> tuple[str, ...]:
    path = resolve_classes_file_path(classes_file, origin="load_class_names")
    labels = _usable_lines(path.read_text(encoding="utf-8-sig"))
    if not labels:
        raise ValueError(f"classes.txt 没有有效类别行: {path}")

    seen: dict[str, str] = {}
    for label in labels:
        for forbidden in _FORBIDDEN_LABEL_PARTS:
            if forbidden in label:
                raise ValueError(
                    f"类别名包含结构化输出保留字符/词 {forbidden!r}: {label!r}"
                )
        key = normalize_lookup_key(label)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"classes.txt 存在重复或规范化后冲突的类别: "
                f"{previous!r} 与 {label!r}"
            )
        seen[key] = label
    return tuple(labels)


def class_file_sha256(classes_file: str | Path) -> str:
    path = resolve_classes_file_path(classes_file, origin="class_file_sha256")
    labels = load_class_names(path)
    payload = json.dumps(
        list(labels), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def multilingual_tokens(value: object) -> list[str]:
    """Tokenize English/number words and Chinese characters for similarity gates."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    latin = re.findall(r"[a-z0-9]+", text)
    han_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text)
    han: list[str] = []
    for run in han_runs:
        # Keep unigrams for short atomic questions and bigrams for semantic overlap.
        han.extend(list(run))
        han.extend(run[i : i + 2] for i in range(max(0, len(run) - 1)))
    return latin + han


def contains_term(text: object, term: object) -> bool:
    """Unicode-safe term containment.

    ASCII labels use token boundaries. CJK/mixed labels use normalized substring
    matching because ``\b`` does not model Chinese word boundaries.
    """
    haystack = normalize_lookup_key(text)
    needle = normalize_lookup_key(term)
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+\-/]*", needle):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
            haystack,
        ) is not None
    return needle in haystack


@dataclass(frozen=True)
class RuntimeCatalog:
    labels: tuple[str, ...]
    aliases: dict[str, str]
    custom: bool
    classes_file: str | None
    sha256: str
    language: str

    def canonicalize(self, value: object) -> str | None:
        key = normalize_lookup_key(value)
        if not key:
            return None
        direct = {normalize_lookup_key(x): x for x in self.labels}
        if key in direct:
            return direct[key]
        alias_target = self.aliases.get(key)
        if alias_target is None:
            return None
        return direct.get(normalize_lookup_key(alias_target))

    def manifest(self) -> dict[str, object]:
        return {
            "mode": "custom" if self.custom else "dota",
            "classes_file": self.classes_file,
            "num_classes": len(self.labels),
            "taxonomy_sha256": self.sha256,
            "qa_language": self.language,
        }


def runtime_catalog() -> RuntimeCatalog:
    classes_file = str(os.getenv("ALS_CLASSES_FILE", "") or "").strip()
    language = str(os.getenv("ALS_QA_LANG", "en") or "en").strip().lower()
    if language not in {"zh", "en"}:
        raise ValueError(f"ALS_QA_LANG 只支持 zh/en，当前为: {language!r}")

    if classes_file:
        path = resolve_classes_file_path(classes_file, origin="ALS_CLASSES_FILE")
        labels = load_class_names(path)
        digest = class_file_sha256(path)
        return RuntimeCatalog(
            labels=labels,
            aliases={},
            custom=True,
            classes_file=str(path),
            sha256=digest,
            language=language,
        )

    payload = json.dumps(
        list(DEFAULT_DOTA_CLASSES), separators=(",", ":")
    ).encode("utf-8")
    aliases = {
        normalize_lookup_key(key): value
        for key, value in DEFAULT_DOTA_ALIASES.items()
    }
    return RuntimeCatalog(
        labels=DEFAULT_DOTA_CLASSES,
        aliases=aliases,
        custom=False,
        classes_file=None,
        sha256=sha256(payload).hexdigest(),
        language=language,
    )


def materialize_runtime_taxonomy(
    settings: dict,
    *,
    classes_file: str | Path | None,
    language: str | None,
) -> RuntimeCatalog:
    """Apply CLI overrides to an already materialized settings dictionary."""
    configured = settings.get("taxonomy") if isinstance(settings.get("taxonomy"), dict) else {}
    requested_file = str(
        classes_file
        or configured.get("classes_file")
        or os.getenv("ALS_CLASSES_FILE", "")
        or ""
    ).strip()
    requested_lang = str(
        language
        or configured.get("qa_language")
        or settings.get("data_conversion", {}).get("qa_language")
        or settings.get("trajectory", {}).get("qa_language")
        or os.getenv("ALS_QA_LANG", "en")
        or "en"
    ).strip().lower()
    if requested_lang not in {"zh", "en"}:
        raise ValueError(f"--lang 只支持 zh 或 en，当前为 {requested_lang!r}")

    old_classes = os.environ.get("ALS_CLASSES_FILE")
    old_lang = os.environ.get("ALS_QA_LANG")
    try:
        if requested_file:
            resolved_file = resolve_classes_file_path(
                requested_file, origin="CLI/settings taxonomy.classes_file"
            )
            os.environ["ALS_CLASSES_FILE"] = str(resolved_file)
        else:
            os.environ.pop("ALS_CLASSES_FILE", None)
        os.environ["ALS_QA_LANG"] = requested_lang
        catalog = runtime_catalog()
    finally:
        if old_classes is None:
            os.environ.pop("ALS_CLASSES_FILE", None)
        else:
            os.environ["ALS_CLASSES_FILE"] = old_classes
        if old_lang is None:
            os.environ.pop("ALS_QA_LANG", None)
        else:
            os.environ["ALS_QA_LANG"] = old_lang

    settings["taxonomy"] = {
        **catalog.manifest(),
        "strict_exact_final_label": True,
        "preserve_output_paths": True,
    }
    settings.setdefault("data_conversion", {})["qa_language"] = requested_lang
    settings.setdefault("trajectory", {})["qa_language"] = requested_lang
    return catalog


def catalog_terms(catalog: RuntimeCatalog | None = None) -> tuple[str, ...]:
    current = catalog or runtime_catalog()
    return tuple(current.labels)
