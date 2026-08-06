from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from openai import OpenAI


def check_server(base_url: str, expected_model: str | None = None, timeout: float = 3.0) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = [str(x.get("id")) for x in payload.get("data", [])]
    if expected_model and expected_model not in models:
        raise RuntimeError(f"Endpoint {url} serves {models}, expected {expected_model}")
    return models


def chat_once(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float = 300,
    retries: int = 1,
) -> dict[str, Any]:
    client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout, max_retries=0)
    started = time.time()
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            return {
                "raw_output": choice.message.content or "",
                "finish_reason": choice.finish_reason,
                "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
                "error": None,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    return {
        "raw_output": "",
        "finish_reason": None,
        "usage": None,
        "error": f"{type(last_error).__name__}: {last_error}",
        "elapsed_seconds": round(time.time() - started, 3),
    }


def protocol(settings: dict[str, Any], name: str) -> dict[str, Any]:
    protocols = settings.get("evaluation", {}).get("protocols", {})
    if name not in protocols:
        raise KeyError(f"Unknown evaluation protocol {name}; available={sorted(protocols)}")
    return protocols[name]


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
