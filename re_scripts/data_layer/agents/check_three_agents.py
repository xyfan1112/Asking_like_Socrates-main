#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main_layer"))
from common import load_settings, settings_from_cli  # noqa: E402


def _json_request(url: str, payload: dict | None = None, timeout: int = 20) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    failed: list[str] = []
    for role, cfg in settings["agents"].items():
        base = cfg["base_url"].rstrip("/")
        try:
            data = _json_request(base + "/models", timeout=8)
            ids = [item.get("id") for item in data.get("data", [])]
            if cfg["served_name"] not in ids:
                raise RuntimeError(
                    f"served_name={cfg['served_name']} not exposed; models={ids}"
                )
            completion = _json_request(
                base + "/chat/completions",
                {
                    "model": cfg["served_name"],
                    "messages": [{"role": "user", "content": "Reply with OK only."}],
                    "temperature": 0.0,
                    "max_tokens": 8,
                },
                timeout=30,
            )
            text = (
                completion.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not str(text).strip():
                raise RuntimeError("chat completion returned empty content")
            print(
                f"[OK] {role}: GPU={cfg['gpu']} URL={cfg['base_url']} "
                f"model={cfg['served_name']} chat={str(text).strip()[:40]!r}"
            )
        except Exception as exc:
            failed.append(role)
            print(f"[FAIL] {role}: {type(exc).__name__}: {exc}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
