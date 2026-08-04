#!/usr/bin/env python3
"""Smoke-test one shared Qwen3-Omni OpenAI-compatible endpoint.

The check performs a model-list request, one text request and one image request.
It never starts or modifies the server.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402


def request_json(url: str, payload: dict | None = None, timeout: int = 120) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def first_image(settings: dict) -> Path:
    root = Path(settings["paths"]["dota128_root"])
    for split in ("val", "train"):
        image_dir = root / split / "images"
        if image_dir.is_dir():
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                found = next(iter(sorted(image_dir.glob(suffix))), None)
                if found:
                    return found
    raise FileNotFoundError(f"no image found under {root}/{{train,val}}/images")


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def text_from_response(value: dict) -> str:
    choices = value.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--base-url")
    ap.add_argument("--served-name")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    shared = settings.get("shared_omni_external", {})
    base = str(args.base_url or shared.get("base_url") or settings["agents"]["reasoner"]["base_url"]).rstrip("/")
    model = str(args.served_name or shared.get("served_name") or settings["agents"]["reasoner"]["served_name"])
    failures: list[str] = []
    report: dict[str, object] = {"base_url": base, "served_name": model}

    try:
        models = request_json(base + "/models", timeout=20)
        ids = [item.get("id") for item in models.get("data", [])]
        report["models"] = ids
        if model not in ids:
            failures.append(f"served_name_not_exposed:{model};models={ids}")
    except Exception as exc:
        failures.append(f"models_request_failed:{type(exc).__name__}:{exc}")

    text_only = bool(shared.get("text_only", False))
    common = {"model": model, "temperature": 0.0, "max_tokens": 32}
    if text_only:
        common["modalities"] = ["text"]
    report["request_modalities_text"] = text_only
    try:
        response = request_json(
            base + "/chat/completions",
            {**common, "messages": [{"role": "user", "content": "只回复OK。"}]},
        )
        text = text_from_response(response)
        report["text_response"] = text[:200]
        if not text:
            failures.append("text_request_empty")
    except Exception as exc:
        failures.append(f"text_request_failed:{type(exc).__name__}:{exc}")

    try:
        image = first_image(settings)
        response = request_json(
            base + "/chat/completions",
            {
                **common,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url(image)}},
                            {"type": "text", "text": "只回答：图像已读取。"},
                        ],
                    }
                ],
            },
        )
        text = text_from_response(response)
        report["image"] = str(image)
        report["image_response"] = text[:200]
        if not text:
            failures.append("image_request_empty")
    except Exception as exc:
        failures.append(f"image_request_failed:{type(exc).__name__}:{exc}")

    report["failures"] = failures
    report["passed"] = not failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[SHARED OMNI ENDPOINT] {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
