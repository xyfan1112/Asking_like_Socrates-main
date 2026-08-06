#!/usr/bin/env python3
"""Custom JSON Socratic engine used only for prompt debugging and fallback.

The scientific mainline remains the unmodified official SocraticAgent generator.
This script mirrors the role split while providing stricter JSON observability and
fast local debugging.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_layer"))
from common import extract_json_object, image_data_url, load_settings, read_jsonl, settings_from_cli, append_jsonl  # noqa: E402
from data_layer.prompts.profiles import get_overlay  # noqa: E402
from openai import OpenAI  # noqa: E402

BASE_REASONER = """You are the Reasoner in a Socratic remote-sensing agent. You cannot see the image. Use the original task and prior visual observations to choose either one new atomic visual question or a final answer. Return JSON only.
Inspect schema: {"thinking":"evidence-based rationale","action":"inspect","visual_question":"one atomic question","draft_answer":""}
Final schema: {"thinking":"synthesis","action":"final","visual_question":"","draft_answer":"answer matching the user format"}
Never repeat a prior question. On the last allowed round choose action=final."""
BASE_PERCEIVER = """You are the Perceiver. You can see the remote-sensing image but not hidden labels. Answer only the current atomic visual question, concisely and without a self-dialogue. Use observable evidence and the coordinate convention stated by the question."""
BASE_VERIFIER = """You are an independent trajectory Verifier. Return JSON only: {"pass":true_or_false,"score":0_to_1,"reason":"brief reason"}. Judge target identity, canonical class, relevance, consistency and whether the final answer follows the requested format. Geometry is also checked by deterministic code later."""


def systems(profile: str) -> tuple[str, str, str]:
    def merge(base: str, role: str) -> str:
        extra = get_overlay(profile, role)
        return base if not extra else f"{base}\n\n{extra}"
    return merge(BASE_REASONER, "reasoner"), merge(BASE_PERCEIVER, "perceiver"), merge(BASE_VERIFIER, "verifier")


def client_call(cfg: dict[str, Any], messages: list[dict[str, Any]], max_tokens: int | None = None, temperature: float | None = None) -> str:
    client = OpenAI(api_key="EMPTY", base_url=cfg["base_url"], timeout=180)
    response = client.chat.completions.create(
        model=cfg["served_name"],
        messages=messages,
        max_tokens=max_tokens or cfg["max_tokens"],
        temperature=cfg["temperature"] if temperature is None else temperature,
        top_p=0.9,
    )
    return response.choices[0].message.content or ""


def parse_legacy(raw: str) -> dict[str, str] | None:
    thinking = re.search(r"<thinking>\s*(.*?)\s*</thinking>", raw, re.I | re.S)
    question = re.search(r"<question>\s*(.*?)\s*</question>", raw, re.I | re.S)
    final = re.search(r"\[Final\s+Answer\]\s*:\s*(.*)", raw, re.I | re.S)
    if question:
        return {"thinking": thinking.group(1).strip() if thinking else "", "action": "inspect", "visual_question": question.group(1).strip(), "draft_answer": ""}
    if final:
        return {"thinking": thinking.group(1).strip() if thinking else "", "action": "final", "visual_question": "", "draft_answer": final.group(1).strip()}
    return None


def parse_reasoner(raw: str) -> dict[str, str] | None:
    value = extract_json_object(raw) or parse_legacy(raw)
    if not isinstance(value, dict):
        return None
    action = str(value.get("action", "")).strip().lower()
    thinking = str(value.get("thinking", "")).strip()
    question = str(value.get("visual_question", "")).strip()
    answer = str(value.get("draft_answer", "")).strip()
    if action not in {"inspect", "final"} or not thinking:
        return None
    if action == "inspect" and not question:
        return None
    if action == "final" and not answer:
        return None
    return {"thinking": thinking, "action": action, "visual_question": question, "draft_answer": answer}


def repair_format(raw: str, cfg: dict[str, Any], reasoner_system: str, require_final: bool) -> tuple[dict[str, str] | None, str]:
    prompt = (
        "Convert the malformed output below into exactly one JSON object matching the required schema. "
        + ("The action must be final. " if require_final else "")
        + f"Preserve meaning.\nMALFORMED:\n{raw}"
    )
    repaired_raw = client_call(
        cfg,
        [{"role": "system", "content": reasoner_system}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    return parse_reasoner(repaired_raw), repaired_raw


def process_one(item: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    trajectory = settings["trajectory"]
    agents = settings["agents"]
    max_rounds = int(trajectory["max_reasoning_rounds"])
    min_rounds = int(trajectory.get("min_perception_rounds_strict", 2))
    profile = trajectory.get("prompt_profile", "official_general")
    reasoner_system, perceiver_system, verifier_system = systems(profile)
    history: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    error = None
    generated_final = ""
    final_thinking = ""

    for round_id in range(1, max_rounds + 1):
        history_text = "\n".join(
            f"Round {h['round']} question: {h['visual_question']}\nObservation: {h['observation']}"
            for h in history
        )
        constraints = []
        if round_id == max_rounds:
            constraints.append("This is the last allowed round. Return action=final.")
        elif len(history) < min_rounds:
            constraints.append(
                f"You have only {len(history)} visual observations; collect at least {min_rounds} before finalizing."
            )
        user = (
            f"Task:\n{item['query']}\n\nKnown observations:\n{history_text or '(none)'}\n\n"
            + " ".join(constraints)
        )
        raw = client_call(
            agents["reasoner"],
            [{"role": "system", "content": reasoner_system}, {"role": "user", "content": user}],
        )
        parsed = parse_reasoner(raw)
        if parsed is None:
            parsed, repair_raw = repair_format(raw, agents["reasoner"], reasoner_system, round_id == max_rounds)
            repairs.append({"round": round_id, "original": raw, "repair_raw": repair_raw, "success": bool(parsed)})
        if not parsed:
            error = "reasoner_format_unrecoverable"
            break
        if parsed["action"] == "final":
            if len(history) < min_rounds and round_id < max_rounds:
                history.append(
                    {
                        "round": round_id,
                        "thinking": parsed["thinking"],
                        "visual_question": "[CONTROL] final answer deferred because minimum observation count was not met",
                        "observation": "The controller requires another independent visual inspection.",
                        "reasoner_raw": raw,
                        "control_round": True,
                    }
                )
                continue
            generated_final = parsed["draft_answer"]
            final_thinking = parsed["thinking"]
            break

        perception_raw = client_call(
            agents["perceiver"],
            [
                {"role": "system", "content": perceiver_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(item["image_path"])}},
                        {"type": "text", "text": parsed["visual_question"]},
                    ],
                },
            ],
        )
        history.append(
            {
                "round": round_id,
                "thinking": parsed["thinking"],
                "visual_question": parsed["visual_question"],
                "observation": perception_raw,
                "reasoner_raw": raw,
                "control_round": False,
            }
        )

    if not generated_final and not error:
        error = "missing_final"

    verifier = {"pass": False, "score": 0.0, "reason": "not_called"}
    if not error:
        trace = "\n".join(
            f"Q: {h['visual_question']}\nO: {h['observation']}" for h in history if not h.get("control_round")
        )
        prompt = f"Task: {item['query']}\nGT: {item['gt']}\nGenerated final: {generated_final}\nTrace:\n{trace}"
        verifier_raw = client_call(
            agents["verifier"],
            [{"role": "system", "content": verifier_system}, {"role": "user", "content": prompt}],
            temperature=0,
        )
        value = extract_json_object(verifier_raw)
        if isinstance(value, dict):
            try:
                score = float(value.get("score", 0.0))
            except Exception:
                score = 0.0
            verifier = {
                "pass": bool(value.get("pass", False)),
                "score": max(0.0, min(1.0, score)),
                "reason": str(value.get("reason", "")),
                "raw": verifier_raw,
            }
        else:
            verifier = {"pass": False, "score": 0.0, "reason": "verifier_invalid_json", "raw": verifier_raw}

    visible_rounds = sum(1 for x in history if not x.get("control_round"))
    return {
        **item,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": "custom_json_fallback_v4",
        "prompt_profile": profile,
        "success": error is None,
        "error": error,
        "rounds": history,
        "perception_rounds": visible_rounds,
        "final_thinking": final_thinking,
        "generated_final": generated_final,
        "format_repairs": repairs,
        "verifier": verifier,
        "teacher_forced_final": item["gt"] if trajectory.get("teacher_force_final_answer", True) else generated_final,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()
    settings = load_settings(settings_from_cli(__file__, args.settings))
    input_path = Path(settings["paths"]["pipeline_work_root"]) / "agent_inputs" / f"{args.split}_agent_inputs.jsonl"
    items = read_jsonl(input_path)
    if args.max_samples:
        items = items[: args.max_samples]
    output = Path(args.output) if args.output else Path(settings["paths"]["trajectory_raw_dir"]) / f"{args.split}_trajectories.jsonl"
    done = set()
    if settings["trajectory"].get("resume", True) and output.exists():
        done = {r["id"] for r in read_jsonl(output, skip_bad=True) if r.get("id")}
    todo = [x for x in items if x["id"] not in done]
    print(f"input={len(items)} done={len(done)} todo={len(todo)} output={output}")
    with ThreadPoolExecutor(max_workers=int(settings["trajectory"]["concurrency"])) as pool:
        futures = {pool.submit(process_one, item, settings): item["id"] for item in todo}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:
                row = {"id": futures[future], "success": False, "error": f"worker_exception:{type(exc).__name__}:{exc}"}
            append_jsonl(output, row)
            print(
                f"[{index}/{len(futures)}] id={row.get('id')} success={row.get('success')} "
                f"rounds={row.get('perception_rounds')} verify={row.get('verifier', {}).get('pass')}"
            )


if __name__ == "__main__":
    main()
