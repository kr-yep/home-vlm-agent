"""Run the agent: `python -m home_vlm --source media/clip.mp4`."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .detect import Detector
from .labeler import TrackLabeler
from .pipeline import Pipeline
from .server import create_app
from .vlm import VLMClient


def main() -> None:
    parser = argparse.ArgumentParser(prog="home_vlm")
    parser.add_argument("--source", default=os.environ.get("HVA_SOURCE", "0"), help="camera index or video file")
    parser.add_argument("--vlm-url", default=os.environ.get("HVA_VLM_URL", "http://vlm:8000"))
    parser.add_argument("--vlm-model", default=os.environ.get("HVA_VLM_MODEL", "gemma-4-12b-it-qat"))
    parser.add_argument("--yolo", default=os.environ.get("HVA_YOLO", "/models/yolo.pt"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8700)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    vlm = VLMClient(args.vlm_url, args.vlm_model)
    labeler = TrackLabeler(vlm)
    pipeline = Pipeline(args.source, Detector(args.yolo), labeler)
    pipeline.start()
    try:
        uvicorn.run(create_app(pipeline, vlm), host=args.host, port=args.port, log_level="warning")
    finally:
        pipeline.stop()
        labeler.close()


if __name__ == "__main__":
    main()
