#!/usr/bin/env python3
"""Small endpoint benchmark for one shared Qwen3-Omni TP=4 server.

It measures end-to-end latency and successful request throughput at several
client concurrency levels. It also samples nvidia-smi during the benchmark so
all four GPUs can be checked for memory allocation and utilization.

This is a deployment benchmark, not a model-quality evaluation.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise TypeError("response root is not JSON object")
    return value


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def first_image(settings: dict[str, Any]) -> Path:
    root = Path(settings["paths"]["dota128_root"])
    for split in ("val", "train"):
        image_dir = root / split / "images"
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            found = next(iter(sorted(image_dir.glob(pattern))), None) if image_dir.is_dir() else None
            if found:
                return found
    raise FileNotFoundError(f"no image under {root}/{{train,val}}/images")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def gpu_snapshot() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "memory_used_mib": float(parts[1]),
                    "memory_total_mib": float(parts[2]),
                    "utilization_gpu_pct": float(parts[3]),
                    "power_w": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return rows


def summarize_gpu(samples: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_gpu: dict[int, dict[str, list[float]]] = {}
    for snapshot in samples:
        for row in snapshot:
            bucket = by_gpu.setdefault(
                int(row["index"]),
                {"memory": [], "util": [], "power": [], "total": []},
            )
            bucket["memory"].append(float(row["memory_used_mib"]))
            bucket["util"].append(float(row["utilization_gpu_pct"]))
            bucket["power"].append(float(row["power_w"]))
            bucket["total"].append(float(row["memory_total_mib"]))
    output = []
    for index in sorted(by_gpu):
        bucket = by_gpu[index]
        output.append(
            {
                "index": index,
                "memory_used_mib_avg": round(statistics.mean(bucket["memory"]), 1),
                "memory_used_mib_max": round(max(bucket["memory"]), 1),
                "memory_total_mib": round(max(bucket["total"]), 1),
                "utilization_gpu_pct_avg": round(statistics.mean(bucket["util"]), 1),
                "utilization_gpu_pct_max": round(max(bucket["util"]), 1),
                "power_w_avg": round(statistics.mean(bucket["power"]), 1),
            }
        )
    return output


def one_request(base_url: str, model: str, image_url: str, timeout: int, index: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 96,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": (
                            "请只输出一行JSON，确认已读取图像并给出图像中最显著目标的大致类别。"
                            f"请求编号={index}。格式：{{\"read\":true,\"object\":\"...\"}}"
                        ),
                    },
                ],
            }
        ],
    }
    start = time.perf_counter()
    response = request_json(base_url.rstrip("/") + "/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - start
    choices = response.get("choices") or []
    content = str(((choices[0] if choices else {}).get("message") or {}).get("content") or "")
    usage = response.get("usage") or {}
    return {
        "ok": bool(content.strip()),
        "latency_s": elapsed,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "finish_reason": (choices[0] if choices else {}).get("finish_reason"),
        "preview": content[:160],
    }


def run_level(
    base_url: str,
    model: str,
    image_url: str,
    concurrency: int,
    request_count: int,
    timeout: int,
) -> dict[str, Any]:
    stop = threading.Event()
    samples: list[list[dict[str, Any]]] = []

    def sample_loop() -> None:
        while not stop.is_set():
            snapshot = gpu_snapshot()
            if snapshot:
                samples.append(snapshot)
            stop.wait(0.5)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(one_request, base_url, model, image_url, timeout, index)
                for index in range(request_count)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        duration = time.perf_counter() - started
        stop.set()
        sampler.join(timeout=2)

    successful = [row for row in results if row.get("ok")]
    latencies = [float(row["latency_s"]) for row in successful]
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in successful)
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "success": len(successful),
        "failure": request_count - len(successful),
        "wall_time_s": round(duration, 3),
        "successful_requests_per_s": round(len(successful) / duration, 4) if duration else 0.0,
        "completion_tokens_per_s": round(completion_tokens / duration, 3) if duration else 0.0,
        "latency_s_mean": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_s_p50": round(percentile(latencies, 0.50), 3) if latencies else None,
        "latency_s_p95": round(percentile(latencies, 0.95), 3) if latencies else None,
        "gpu": summarize_gpu(samples),
        "errors": [row.get("error") for row in results if row.get("error")][:10],
        "previews": [row.get("preview") for row in successful[:2]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--requests-per-level", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    image = first_image(settings)
    image_url = data_url(image)
    levels = []
    for part in args.concurrency.split(","):
        value = int(part.strip())
        if value < 1:
            raise ValueError("concurrency must be >=1")
        levels.append(value)

    report: dict[str, Any] = {
        "base_url": args.base_url,
        "served_name": args.served_name,
        "image": str(image),
        "levels": [],
    }
    for level in levels:
        print(f"[BENCH] concurrency={level} requests={args.requests_per_level}", flush=True)
        row = run_level(
            args.base_url,
            args.served_name,
            image_url,
            level,
            args.requests_per_level,
            args.timeout,
        )
        report["levels"].append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    failures = sum(int(row["failure"]) for row in report["levels"])
    report["passed"] = failures == 0
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[REPORT]", args.output)
    print(f"[OMNI TP4 BENCHMARK] {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
