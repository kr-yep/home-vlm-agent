"""YOLO as a box proposer.

YOLO's class names are only a hint for the VLM. What it is good at is finding
and tracking boxes cheaply on every frame, so the confidence threshold is kept
low: a weak box the VLM can name beats a missed object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int | None
    yolo_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels


class Detector:
    def __init__(self, weights: str, *, confidence: float = 0.2, imgsz: int = 960, device: str | int = 0):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._confidence = confidence
        self._imgsz = imgsz
        self._device = device

    def track(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self._model.track(
            frame_bgr,
            persist=True,
            conf=self._confidence,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )[0]
        return self._detections(result)

    def detect(self, image) -> list[Detection]:
        """Same settings as `track`, for unrelated still images (no track ids)."""

        result = self._model.predict(image, conf=self._confidence, imgsz=self._imgsz, device=self._device, verbose=False)[0]
        return self._detections(result)

    @staticmethod
    def _detections(result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        return [
            Detection(
                track_id=track_id,
                yolo_name=result.names[int(cls)],
                confidence=float(conf),
                box=tuple(int(v) for v in xyxy),
            )
            for track_id, cls, conf, xyxy in zip(ids, boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())
        ]
