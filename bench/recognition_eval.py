"""Name each evaluation object with YOLO alone and with YOLO + VLM.

For every LVIS object YOLO runs on the whole image (the app's settings) and the
detection with the highest IoU (>= 0.5) is its YOLO box. The VLM then names
  - gt_nohint:   the ground-truth box, no YOLO guess   (best case for the VLM)
  - yolo_hint:   the YOLO box with YOLO's guess        (what the app does)
  - yolo_nohint: the YOLO box without the guess        (does the guess anchor it?)
Scoring is a separate step (`bench/score.py`). Run on the internal network:
    docker compose run --rm --no-deps -v /root/.cache/home-vlm/data:/data app \
        python -m bench.recognition_eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from home_vlm.detect import Detector
from home_vlm.labeler import crop_with_margin, mark_box
from home_vlm.vlm import VLMClient


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/lvis-eval")
    parser.add_argument("--out", default="/runs/recognition.jsonl")
    parser.add_argument("--vlm-url", default="http://vlm:8000")
    parser.add_argument("--vlm-model", default="gemma-4-12b-it-qat")
    parser.add_argument("--yolo", default="/models/yolo.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--view", choices=["crop", "marked"], default="crop", help="tight crop, or wider view with the box drawn in red")
    args = parser.parse_args()

    data = Path(args.data)
    objects = json.loads((data / "manifest.json").read_text())
    if args.limit:
        objects = objects[: args.limit]
    detector = Detector(args.yolo)
    vlm = VLMClient(args.vlm_url, args.vlm_model)
    detections_by_image: dict[int, list] = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Resumable: objects already in the output are skipped, so an interrupted run continues.
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()} if out.exists() else set()
    started = time.perf_counter()
    with out.open("a", encoding="utf-8") as sink:
        for n, obj in enumerate(objects, 1):
            if obj["id"] in done:
                continue
            image = Image.open(data / "images" / obj["file"]).convert("RGB")
            if obj["image_id"] not in detections_by_image:
                detections_by_image[obj["image_id"]] = [(d.box, d.yolo_name, d.confidence) for d in detector.detect(image)]
            gt = tuple(obj["bbox_xyxy"])
            best = max(detections_by_image[obj["image_id"]], key=lambda d: iou(d[0], gt), default=None)
            matched = best is not None and iou(best[0], gt) >= 0.5
            record = {
                "id": obj["id"],
                "category": obj["category"],
                "split": obj["split"],
                "yolo_class": best[1] if matched else None,
                "yolo_conf": best[2] if matched else None,
                "yolo_iou": iou(best[0], gt) if best else 0.0,
            }
            runs = [("gt_nohint", gt, None)]
            if matched:
                runs += [("yolo_hint", best[0], best[1]), ("yolo_nohint", best[0], None)]
            for key, box, hint in runs:
                try:
                    if args.view == "marked":
                        label = vlm.describe(mark_box(image, box), hint, marked=True)
                    else:
                        label = vlm.describe(crop_with_margin(image, box), hint)
                    record[key] = {
                        "name": label.name,
                        "name_en": label.name_en,
                        "detail": label.detail,
                        "confident": label.confident,
                        "latency_s": round(label.latency_s, 3),
                        "parsed": bool(label.name_en),
                    }
                except Exception as error:  # keep going; a failure is a result
                    record[key] = {"error": repr(error)}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            if n % 25 == 0:
                print(f"{n}/{len(objects)} {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done {len(objects)} objects in {time.perf_counter() - started:.0f}s -> {out}")


if __name__ == "__main__":
    main()
