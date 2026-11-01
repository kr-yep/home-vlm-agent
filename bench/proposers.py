"""Which detector finds the most objects for the VLM to name? Boxes only, no VLM.

Every detector runs on the evaluation images at a very low threshold, class-agnostic
NMS on, and its boxes are kept. A detector that only finds more by proposing more is
not better, since each new box costs a VLM call, so recall is reported against the
number of boxes per image (AR@K, the usual proposal metric, here at IoU 0.5 only).
LVIS labels are incomplete, so a box with no label is not necessarily wrong and
precision is not measured.
    docker compose run --rm --no-deps -v /root/.cache/home-vlm/data:/data app \
        python -m bench.proposers /models/yolo26m.pt /models/yoloe-26m-seg-pf.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from .recognition_eval import iou

BUDGETS = (5, 10, 20, 50, 100)
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)


def run(weights: str, images: dict[int, str], data: Path, imgsz: int) -> tuple[dict[int, list], float]:
    from ultralytics import YOLO

    model = YOLO(weights)
    boxes: dict[int, list] = {}
    times = []
    for image_id, name in images.items():
        image = Image.open(data / "images" / name).convert("RGB")
        started = time.perf_counter()
        result = model.predict(image, conf=0.01, imgsz=imgsz, max_det=300, agnostic_nms=True, device=0, verbose=False)[0]
        times.append(time.perf_counter() - started)
        found = zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        boxes[image_id] = sorted(([xyxy, conf, result.names[int(cls)]] for xyxy, conf, cls in found), key=lambda b: -b[1])
    return boxes, statistics.median(times[5:]) * 1000  # the first few include warm-up


def recall(objects: list[dict], boxes: dict[int, list], keep) -> dict[str, float]:
    found = {"coco": [], "non_coco": []}
    for obj in objects:
        kept = keep(boxes[obj["image_id"]])
        found[obj["split"]].append(any(iou(b[0], obj["bbox_xyxy"]) >= 0.5 for b in kept))
    every = found["coco"] + found["non_coco"]
    return {split: round(sum(v) / len(v), 3) for split, v in {**found, "all": every}.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="+")
    parser.add_argument("--data", default="/data/lvis-eval")
    parser.add_argument("--out", default="/runs/proposers.json")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    data = Path(args.data)
    objects = json.loads((data / "manifest.json").read_text())
    images = {o["image_id"]: o["file"] for o in objects}
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {}
    for weights in args.weights:
        boxes, ms = run(weights, images, data, args.imgsz)
        report[Path(weights).stem] = {
            "ms_per_image_median": round(ms, 1),
            "at_top_k": {k: recall(objects, boxes, lambda b, k=k: b[:k]) for k in BUDGETS},
            "at_confidence": {
                t: {
                    "boxes_per_image": round(statistics.mean(sum(b[1] >= t for b in v) for v in boxes.values()), 1),
                    **recall(objects, boxes, lambda b, t=t: [x for x in b if x[1] >= t]),
                }
                for t in THRESHOLDS
            },
        }
        out.write_text(json.dumps(report, indent=1))
        print(Path(weights).stem, json.dumps(report[Path(weights).stem]), flush=True)


if __name__ == "__main__":
    main()
