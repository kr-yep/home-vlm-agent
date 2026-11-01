"""Latency of one runtime on the app's real workload: one crop at a time.

Sends the same label request the app sends (same prompt, image size, max_tokens,
temperature 0, JSON mode), streamed so time-to-first-token is visible, while
YOLO runs on the GPU in the same process the way the app does. Then measures
chat latency with the GPU idle and while labelling keeps the model busy.
    docker compose run --rm --no-deps -v /root/.cache/home-vlm/data:/data app \
        python -m bench.runtime_bench --name llamacpp --url http://llamacpp:8080 --model gemma-4-12b-it-qat
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import httpx
from PIL import Image

from home_vlm.detect import Detector
from home_vlm.labeler import crop_with_margin
from home_vlm.vlm import CHAT_SYSTEM, DESCRIBE_HINT, DESCRIBE_SYSTEM, DESCRIBE_USER, encode_image

QUESTIONS = ["畫面裡有什麼？", "這個東西是做什麼用的？", "它是什麼顏色？", "桌上有杯子嗎？", "最大的那個物品是什麼？"]


def stream(client: httpx.Client, url: str, body: dict) -> dict:
    started = time.perf_counter()
    first = None
    text, usage, timings = [], None, None
    with client.stream("POST", url, json=body | {"stream": True, "stream_options": {"include_usage": True}}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
    total = time.perf_counter() - started
    completion = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n") or 0
    ttft = (first - started) if first else total
    return {
        "ttft_s": ttft,
        "total_s": total,
        "prompt_tokens": (usage or {}).get("prompt_tokens") or (timings or {}).get("prompt_n"),
        "completion_tokens": completion,
        "decode_tok_s": (completion - 1) / (total - ttft) if completion > 1 and total > ttft else None,
        "text": "".join(text),
    }


def label_body(model: str, crop: Image.Image, hint: str, extra: dict) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 160,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": DESCRIBE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(crop)}},
                    {"type": "text", "text": DESCRIBE_HINT.format(hint=hint) + DESCRIBE_USER},
                ],
            },
        ],
    } | extra


def chat_body(model: str, crop: Image.Image, question: str, extra: dict) -> dict:
    content = [
        {"type": "image_url", "image_url": {"url": encode_image(crop)}},
        {"type": "text", "text": f"目前畫面中的物品：\n- 白色馬克杯（YOLO：cup）\n- 筆記型電腦（YOLO：laptop）\n\n問題：{question}"},
    ]
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": content}],
    } | extra


def summary(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}
    return {
        "median": round(statistics.median(values), 3),
        "p95": round(values[max(0, int(round(0.95 * len(values))) - 1)], 3),
        "min": round(values[0], 3),
        "max": round(values[-1], 3),
    }


def chat_images(objects: list[dict], data: Path, n: int) -> list[Image.Image]:
    images = []
    for obj in objects[:n]:
        image = Image.open(data / "images" / obj["file"]).convert("RGB")
        images.append(crop_with_margin(image, tuple(obj["bbox_xyxy"])))
    return images


class YoloLoad(threading.Thread):
    """Run YOLO at the app's frame rate on the same GPU."""

    def __init__(self, weights: str, frame: Image.Image, fps: float = 15.0):
        super().__init__(daemon=True)
        self.detector = Detector(weights)
        self.frame = frame
        self.period = 1.0 / fps
        self.stop = threading.Event()
        self.ms: list[float] = []

    def run(self) -> None:
        while not self.stop.is_set():
            started = time.perf_counter()
            self.detector.detect(self.frame)
            self.ms.append((time.perf_counter() - started) * 1000)
            self.stop.wait(max(0.0, self.period - (time.perf_counter() - started)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="/data/lvis-eval")
    parser.add_argument("--yolo", default="/models/yolo.pt")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--extra", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    parser.add_argument("--out", default="/runs/runtime")
    args = parser.parse_args()
    extra = json.loads(args.extra)
    url = args.url.rstrip("/") + "/v1/chat/completions"

    data = Path(args.data)
    objects = json.loads((data / "manifest.json").read_text())
    crops = []
    for obj in objects[: args.warmup + args.requests]:
        image = Image.open(data / "images" / obj["file"]).convert("RGB")
        crops.append((crop_with_margin(image, tuple(obj["bbox_xyxy"])), obj["category"]))
    frame = Image.open(data / "images" / objects[0]["file"]).convert("RGB")

    client = httpx.Client(timeout=120)
    yolo = YoloLoad(args.yolo, frame)
    yolo.start()
    time.sleep(2)

    for crop, _ in crops[: args.warmup]:
        stream(client, url, label_body(args.model, crop, "cup", extra))
    labels = []
    for crop, category in crops[args.warmup :]:
        result = stream(client, url, label_body(args.model, crop, "cup", extra))
        result["category"] = category
        try:
            result["json_ok"] = isinstance(json.loads(result["text"]), dict)
        except json.JSONDecodeError:
            result["json_ok"] = False
        labels.append(result)

    # Chat images come from a set the label requests never use, and the busy run uses
    # different ones again, so no runtime can answer from its prefix cache.
    chat_crops = chat_images(objects[args.warmup + args.requests :], data, 20)
    chat_idle = [stream(client, url, chat_body(args.model, chat_crops[i], q, extra)) for i, q in enumerate(QUESTIONS * 2)]

    # Labelling keeps going in the background, the way TrackLabeler's one worker does.
    busy = threading.Event()
    busy.set()

    def keep_labelling():
        own = httpx.Client(timeout=120)
        i = 0
        while busy.is_set():
            stream(own, url, label_body(args.model, crops[i % len(crops)][0], "cup", extra))
            i += 1

    worker = threading.Thread(target=keep_labelling, daemon=True)
    worker.start()
    time.sleep(1)
    chat_busy = [stream(client, url, chat_body(args.model, chat_crops[10 + i], q, extra)) for i, q in enumerate(QUESTIONS * 2)]
    busy.clear()
    worker.join(timeout=120)
    yolo.stop.set()

    report = {
        "runtime": args.name,
        "model": args.model,
        "requests": len(labels),
        "label_ttft_s": summary([r["ttft_s"] for r in labels]),
        "label_total_s": summary([r["total_s"] for r in labels]),
        "label_decode_tok_s": summary([r["decode_tok_s"] for r in labels]),
        "label_prompt_tokens": summary([r["prompt_tokens"] for r in labels]),
        "label_completion_tokens": summary([r["completion_tokens"] for r in labels]),
        "label_json_ok": f"{sum(r['json_ok'] for r in labels)}/{len(labels)}",
        "chat_idle_total_s": summary([r["total_s"] for r in chat_idle]),
        "chat_idle_ttft_s": summary([r["ttft_s"] for r in chat_idle]),
        "chat_while_labelling_total_s": summary([r["total_s"] for r in chat_busy]),
        "chat_while_labelling_ttft_s": summary([r["ttft_s"] for r in chat_busy]),
        "yolo_ms_during_run": summary(yolo.ms),
        "sample_label": labels[0]["text"][:300],
        "sample_chat": chat_idle[0]["text"][:300],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"report": report, "labels": labels, "chat_idle": chat_idle, "chat_busy": chat_busy}, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
