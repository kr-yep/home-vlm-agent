"""Run the agent: `python -m home_vlm --source media/clip.mp4`."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .detect import Detector
from .labeler import TrackLabeler
from .pipeline import BrowserCamera, Pipeline
from .server import create_app
from .vlm import VLMClient


def main() -> None:
    parser = argparse.ArgumentParser(prog="home_vlm")
    parser.add_argument("--source", default=os.environ.get("HVA_SOURCE", "browser"), help="browser (the page's camera), a camera index or a video file")
    parser.add_argument("--vlm-url", default=os.environ.get("HVA_VLM_URL", "http://vlm:8000"))
    parser.add_argument("--vlm-model", default=os.environ.get("HVA_VLM_MODEL", "gemma-4-26b-a4b"))
    parser.add_argument("--yolo", default=os.environ.get("HVA_YOLO", "/models/yoloe-26l-o365.pt"))
    parser.add_argument("--confidence", type=float, default=float(os.environ.get("HVA_CONF", "0.3")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("HVA_VLM_WORKERS", "3")), help="naming requests at once; leave a server slot for chat")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8700)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    vlm = VLMClient(args.vlm_url, args.vlm_model)
    labeler = TrackLabeler(vlm, workers=args.workers, max_pending=args.workers)
    camera = BrowserCamera() if args.source == "browser" else None
    pipeline = Pipeline(camera or args.source, Detector(args.yolo, confidence=args.confidence), labeler)
    pipeline.start()
    try:
        uvicorn.run(create_app(pipeline, vlm, camera), host=args.host, port=args.port, log_level="warning")
    finally:
        pipeline.stop()
        labeler.close()


if __name__ == "__main__":
    main()
