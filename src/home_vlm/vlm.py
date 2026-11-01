"""Client for a local vision-language model behind an OpenAI-compatible chat API.

Both llama.cpp's llama-server and Ollama speak this API, so the runtime is a
URL, not a code path.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field

import httpx
from PIL import Image

from .zh import to_taiwan

DESCRIBE_SYSTEM = "你是物品辨識助手。使用者給你攝影機畫面裡一個物體的裁切圖。只輸出 JSON，不要其他文字。"
# YOLO's guess, when given; it is often wrong, so the model is told to overrule it.
DESCRIBE_HINT = "物體偵測器猜是「{hint}」，但它常認錯，請以你自己看到的為準。"
DESCRIBE_JSON = (
    '輸出 JSON：{"name": "繁體中文物品名稱，越具體越好", "name_en": "short English noun", '
    '"detail": "一句外觀描述：顏色、材質、品牌或特徵", "confident": true 或 false}'
)
DESCRIBE_USER = "這個框裡主要的物體是什麼？" + DESCRIBE_JSON
# For a wider view with the object outlined in red (labeler.mark_box): the model
# then knows which object is meant, not just whatever is most eye-catching.
DESCRIBE_USER_MARKED = "圖中紅色框起來的那個物體是什麼？只回答紅框框住的那一個，不是它上面、旁邊或背景的東西。" + DESCRIBE_JSON
CHAT_SYSTEM = (
    "你是住家裡的物品助手，用自然的台灣繁體中文簡短回答。"
    "你看得到附上的畫面，下面也列出已辨識的物品；兩邊都沒有的東西就說沒看到，不要猜。問到用途時，依物品的常見用途回答。"
)

# Crops are tiny next to a full frame; scaling them up a little helps the vision
# encoder, scaling them past this only costs image tokens.
CROP_MIN_SIDE = 224
CROP_MAX_SIDE = 512


@dataclass
class Label:
    name: str
    name_en: str = ""
    detail: str = ""
    confident: bool = False
    latency_s: float = 0.0
    raw: str = field(default="", repr=False)


def encode_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    scale = 1.0
    short, long = min(image.size), max(image.size)
    if short < CROP_MIN_SIDE:
        scale = CROP_MIN_SIDE / short
    if long * scale > CROP_MAX_SIDE:
        scale = CROP_MAX_SIDE / long
    if scale != 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_label(raw: str) -> Label:
    """Read the model's JSON; fall back to its plain text rather than failing."""

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not str(data.get("name", "")).strip():
        text = raw.strip().splitlines()[0] if raw.strip() else "（無法辨識）"
        return Label(name=to_taiwan(text[:40]), raw=raw)
    return Label(
        name=to_taiwan(str(data["name"]).strip()),
        name_en=str(data.get("name_en", "")).strip(),
        detail=to_taiwan(str(data.get("detail", "")).strip()),
        confident=data.get("confident") is True,
        raw=raw,
    )


class VLMClient:
    def __init__(self, base_url: str, model: str, *, timeout_s: float = 60.0, extra_body: dict | None = None):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        # Gemma 4 thinks by default; a label does not need it and it triples latency.
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}} | (extra_body or {})
        self._http = httpx.Client(timeout=timeout_s)

    def _post(self, messages: list[dict], *, max_tokens: int, json_mode: bool) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            **self.extra_body,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        response = self._http.post(self.url, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def describe(self, crop: Image.Image, hint: str | None = None, *, marked: bool = False) -> Label:
        started = time.perf_counter()
        prompt = (DESCRIBE_HINT.format(hint=hint) if hint else "") + (DESCRIBE_USER_MARKED if marked else DESCRIBE_USER)
        raw = self._post(
            [
                {"role": "system", "content": DESCRIBE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_image(crop)}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=160,
            json_mode=True,
        )
        label = parse_label(raw)
        label.latency_s = time.perf_counter() - started
        return label

    def chat(self, question: str, scene: str, images: list[Image.Image] = ()) -> str:
        content = [{"type": "image_url", "image_url": {"url": encode_image(image)}} for image in images]
        content.append({"type": "text", "text": f"目前畫面中的物品：\n{scene}\n\n問題：{question}"})
        raw = self._post(
            [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": content}],
            max_tokens=300,
            json_mode=False,
        )
        return to_taiwan(raw.strip())
