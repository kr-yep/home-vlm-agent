"""Frame loop: read a source, track boxes with YOLO, overlay VLM names.

The source is a camera index or a video file. A file is replayed at its own
frame rate and looped, so a recording behaves like a camera.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass

import cv2
from PIL import Image, ImageDraw, ImageFont

from .detect import Detection, Detector
from .labeler import TrackLabeler

log = logging.getLogger(__name__)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FIRST_FRAME_TIMEOUT_S = 5.0  # a camera can fail its first few reads


@dataclass(frozen=True)
class SeenObject:
    track_id: int | None
    yolo_name: str
    yolo_confidence: float
    box: tuple[int, int, int, int]
    name: str | None
    detail: str | None

    def describe(self) -> str:
        return f"- {self.name}（{self.detail or '無描述'}）"


def load_font(frame_width: int) -> ImageFont.ImageFont:
    size = max(12, frame_width // 45)
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        log.warning("CJK font missing at %s; labels will not render Chinese", FONT_PATH)
        return ImageFont.load_default(size)


def open_source(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    return capture


def draw(frame: Image.Image, objects: list[SeenObject], font: ImageFont.ImageFont) -> Image.Image:
    canvas = frame.copy()
    pen = ImageDraw.Draw(canvas)
    for obj in objects:
        named = obj.name is not None
        colour = (40, 200, 120) if named else (240, 180, 40)
        pen.rectangle(obj.box, outline=colour, width=3)
        text = obj.name if named else f"? {obj.yolo_name}"
        x, y = obj.box[0], max(0, obj.box[1] - font.size - 6)
        pen.rectangle(pen.textbbox((x, y), text, font=font, stroke_width=0), fill=(0, 0, 0))
        pen.text((x, y), text, font=font, fill=colour)
    return canvas


class Pipeline(threading.Thread):
    def __init__(self, source: str, detector: Detector, labeler: TrackLabeler, *, loop: bool = True):
        super().__init__(daemon=True, name="pipeline")
        self._source = source
        self._detector = detector
        self._labeler = labeler
        self._loop = loop
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: Image.Image | None = None
        self._jpeg: bytes | None = None
        self._objects: list[SeenObject] = []
        self.frames = 0
        self.yolo_ms = 0.0
        self.error: str | None = None  # set if the frame loop dies, so the API can say so
        self._font: ImageFont.ImageFont | None = None

    def run(self) -> None:
        try:
            self._run()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            log.exception("frame loop stopped")

    def _run(self) -> None:
        capture = open_source(self._source)
        interval = 1.0 / (capture.get(cv2.CAP_PROP_FPS) or 15.0)
        opened = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                ok, frame_bgr = capture.read()
                if not ok:
                    if self.frames == 0:
                        if time.monotonic() - opened > FIRST_FRAME_TIMEOUT_S:
                            raise RuntimeError(f"cannot read any frame from {self._source!r}; unsupported codec?")
                        time.sleep(0.05)
                        continue
                    if not self._loop:
                        break
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.step(frame_bgr)
                time.sleep(max(0.0, interval - (time.perf_counter() - started)))
        finally:
            capture.release()

    def step(self, frame_bgr) -> list[SeenObject]:
        started = time.perf_counter()
        # A box too small to name would stay "unnamed" for ever, on screen and in chat.
        detections = [d for d in self._detector.track(frame_bgr) if self._labeler.can_name(d)]
        self.yolo_ms = (time.perf_counter() - started) * 1000
        frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._labeler.update(frame, detections)
        objects = [self._seen(detection) for detection in detections]
        self._font = self._font or load_font(frame.width)
        buffer = io.BytesIO()
        draw(frame, objects, self._font).save(buffer, format="JPEG", quality=80)
        with self._lock:
            self._frame, self._objects, self._jpeg = frame, objects, buffer.getvalue()
            self.frames += 1
        return objects

    def _seen(self, detection: Detection) -> SeenObject:
        label = self._labeler.label(detection.track_id)
        return SeenObject(
            track_id=detection.track_id,
            yolo_name=detection.yolo_name,
            yolo_confidence=round(detection.confidence, 2),
            box=detection.box,
            name=label.name if label else None,
            detail=label.detail if label else None,
        )

    def snapshot(self) -> tuple[Image.Image | None, list[SeenObject], bytes | None]:
        with self._lock:
            return self._frame, list(self._objects), self._jpeg

    def stop(self) -> None:
        self._stop.set()
