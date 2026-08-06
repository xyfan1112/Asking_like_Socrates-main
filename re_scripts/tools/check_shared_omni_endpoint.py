#!/usr/bin/env python3
"""Smoke-test one shared Qwen3-Omni AWQ-No-TTS TP=4 endpoint.

Checks /models, a Reasoner-style text call, a Perceiver-style image call, and a
Verifier-style structured text call. It never starts or modifies the server.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise TypeError("response root is not object")
    return value


def first_image(settings: dict[str, Any]) -> Path:
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


def parse_choice(value: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    choices = value.get("choices") or []
    if not choices:
        return "", None, value.get("usage") or {}
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        text = "".join(parts).strip()
    else:
        text = str(content or "").strip()
    return text, choice.get("finish_reason"), value.get("usage") or {}


def run_call(base: str, payload: dict[str, Any], label: str, failures: list[str], report: dict[str, Any]) -> None:
    try:
        response = request_json(base + "/chat/completions", payload)
        text, finish, usage = parse_choice(response)
        report[label] = {
            "text": text[:500],
            "finish_reason": finish,
            "usage": usage,
        }
        if not text:
            failures.append(f"{label}_empty")
        if str(finish).lower() == "length":
            failures.append(f"{label}_finish_reason_length")
    except Exception as exc:
        failures.append(f"{label}_failed:{type(exc).__name__}:{exc}")


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
    report: dict[str, Any] = {
        "base_url": base,
        "served_name": model,
        "shared_contract": shared,
    }

    try:
        models = request_json(base + "/models", timeout=30)
        ids = [item.get("id") for item in models.get("data", [])]
        report["models"] = ids
        if model not in ids:
            failures.append(f"served_name_not_exposed:{model};models={ids}")
    except Exception as exc:
        failures.append(f"models_request_failed:{type(exc).__name__}:{exc}")

    common: dict[str, Any] = {"model": model, "temperature": 0.0, "max_tokens": 96}
    run_call(
        base,
        {
            **common,
            "messages": [
                {
                    "role": "system",
                    "content": "你是Reasoner，只按要求输出结构化文本。",
                },
                {
                    "role": "user",
                    "content": "只输出：<thinking>已理解</thinking><question>图像中是否存在目标？</question>",
                },
            ],
        },
        "reasoner_text",
        failures,
        report,
    )

    try:
        image = first_image(settings)
        report["image"] = str(image)
        run_call(
            base,
            {
                **common,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是Perceiver，只描述图像中直接可见的事实。",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url(image)}},
                            {"type": "text", "text": "只回答：图像已读取，并用不超过20个字描述最显著目标。"},
                        ],
                    },
                ],
            },
            "perceiver_image",
            failures,
            report,
        )
    except Exception as exc:
        failures.append(f"image_prepare_failed:{type(exc).__name__}:{exc}")

    run_call(
        base,
        {
            **common,
            "messages": [
                {
                    "role": "system",
                    "content": "你是Verifier，只输出ACCEPT或REJECT。",
                },
                {
                    "role": "user",
                    "content": "候选答案严格等于GT。只输出ACCEPT。",
                },
            ],
        },
        "verifier_text",
        failures,
        report,
    )

    report["failures"] = failures
    report["passed"] = not failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[SHARED OMNI ENDPOINT] {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
