#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
替换仓库根目录 utils.py 后，官方 SocraticAgent/generation.py 无需改动：
- 代码中的 gpt-5-mini 自动路由到本地 Reasoner vLLM；
- 代码中的 gemini-2.5-flash 自动路由到本地 Perceiver vLLM；
- 全程只访问 127.0.0.1，不调用外网 API。
"""

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image


def load_settings():
    settings_path = Path(
        os.environ.get(
            "FINED_SETTINGS",
            "/home/yk/Asking_like_Socrates/fined_scripts/settings.json",
        )
    )

    with settings_path.open("r", encoding="utf-8") as f:
        return json.load(f)


SETTINGS = load_settings()

RUNTIME_ENV = Path(
    "/home/yk/Asking_like_Socrates/fined_scripts/local_agent_runtime.env"
)

if RUNTIME_ENV.is_file():
    for raw_line in RUNTIME_ENV.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


REASONER_BASE_URL = os.environ.get(
    "LOCAL_REASONER_BASE_URL",
    SETTINGS["reasoner_base_url"],
)

PERCEIVER_BASE_URL = os.environ.get(
    "LOCAL_PERCEIVER_BASE_URL",
    SETTINGS["perceiver_base_url"],
)

REASONER_MODEL = os.environ.get(
    "LOCAL_REASONER_MODEL",
    SETTINGS["reasoner_served_name"],
)

PERCEIVER_MODEL = os.environ.get(
    "LOCAL_PERCEIVER_MODEL",
    SETTINGS["perceiver_served_name"],
)

MAX_RETRIES = int(
    os.environ.get("LOCAL_AGENT_MAX_RETRIES", "4")
)


def encode_image(image: Image.Image, suffix: str) -> str:
    suffix = suffix.lower().replace("jpg", "jpeg")
    buffer = BytesIO()
    image.save(buffer, format=suffix.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_messages(query, images, system_prompt):
    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    if not images:
        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )
        return messages

    content = []

    for image, suffix in images:
        suffix = suffix.lower().replace("jpg", "jpeg")
        encoded = encode_image(image, suffix)

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{suffix};base64,{encoded}"
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": query,
        }
    )

    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )

    return messages


class APIModel:
    def __init__(self, model_name, system_prompt=None):
        self.original_model_name = model_name
        self.system_prompt = system_prompt

        if model_name.startswith(("gpt-", "o")):
            self.base_url = REASONER_BASE_URL
            self.served_model = REASONER_MODEL
            self.max_tokens = int(
                SETTINGS.get("reasoner_max_tokens", 512)
            )
            self.temperature = 0.15

        elif model_name.startswith("gemini-"):
            self.base_url = PERCEIVER_BASE_URL
            self.served_model = PERCEIVER_MODEL
            self.max_tokens = int(
                SETTINGS.get("perceiver_max_tokens", 384)
            )
            self.temperature = 0.10

        elif model_name.startswith(("doubao-", "seed-")):
            # 本地 verifier 与 Reasoner 共用文本模型。
            self.base_url = REASONER_BASE_URL
            self.served_model = REASONER_MODEL
            self.max_tokens = int(
                SETTINGS.get("verifier_max_tokens", 256)
            )
            self.temperature = 0.0

        else:
            raise NotImplementedError(
                f"未配置本地模型映射：{model_name}"
            )

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
            timeout=180.0,
        )

    def get_response(self, query, images=None):
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.served_model,
                    messages=build_messages(
                        query,
                        images,
                        self.system_prompt,
                    ),
                    temperature=self.temperature,
                    top_p=0.90,
                    max_tokens=self.max_tokens,
                )

                content = response.choices[0].message.content

                if content is None:
                    raise RuntimeError("本地模型返回空内容")

                return content

            except Exception as exc:
                last_error = exc

                print(
                    f"[LOCAL API RETRY] "
                    f"{self.original_model_name} -> "
                    f"{self.served_model} "
                    f"{attempt}/{MAX_RETRIES}: {exc}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(min(8.0, 1.5 ** attempt))

        raise last_error
