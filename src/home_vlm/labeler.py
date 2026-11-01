"""Ask the VLM about each tracked object once, without holding up the frame loop.

YOLO runs on every frame; the VLM takes hundreds of milliseconds per crop. So a
track is sent to the VLM only after it has been seen for a few frames (a
one-frame flicker is not worth a model call), only once, and on background
workers, oldest track first. A slow model therefore delays labels, never frames. A track
whose box changes size a lot has probably drifted to something else, so its name
is dropped and asked for again.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image, ImageDraw

from .detect import Detection
from .vlm import Label

log = logging.getLogger(__name__)


class Describer(Protocol):
    def describe(self, crop: Image.Image, hint: str) -> Label: ...


@dataclass
class _Track:
    yolo_name: str
    first_seen: float
    last_seen: float
    frames: int = 0
    label: Label | None = None
    pending: Future | None = field(default=None, repr=False)
    failed: int = 0
    asked_area: int = 0


def crop_with_margin(frame: Image.Image, box: tuple[int, int, int, int], margin: float = 0.1) -> Image.Image:
    """A little context around the box helps the model tell a cup from a vase."""

    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    return frame.crop(
        (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(frame.width, int(x2 + pad_x)),
            min(frame.height, int(y2 + pad_y)),
        )
    )


def mark_box(frame: Image.Image, box: tuple[int, int, int, int], context: float = 1.0) -> Image.Image:
    """The box outlined in red, with `context` x its size of surroundings on each side."""

    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * context, (y2 - y1) * context
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    view = frame.crop((left, top, min(frame.width, int(x2 + pad_x)), min(frame.height, int(y2 + pad_y)))).convert("RGB")
    width = max(2, round(max(view.size) / 150))
    ImageDraw.Draw(view).rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=(255, 0, 0), width=width)
    return view


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


class TrackLabeler:
    def __init__(
        self,
        describer: Describer,
        *,
        min_frames: int = 15,
        min_side: int = 24,
        max_pending: int = 2,
        forget_after_s: float = 10.0,
        max_failures: int = 2,
        workers: int = 1,
        max_size_change: float = 4.0,
    ):
        self._describer = describer
        self._min_frames = min_frames
        self._min_side = min_side
        self._max_pending = max_pending
        self._forget_after_s = forget_after_s
        self._max_failures = max_failures
        self._max_size_change = max_size_change
        self._tracks: dict[int, _Track] = {}
        # Re-entrant: a future that is already done runs its callback inside submit's caller.
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm")

    def update(self, frame: Image.Image, detections: list[Detection], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            for detection in sorted(detections, key=self._priority, reverse=True):
                if detection.track_id is None:
                    continue
                track = self._tracks.setdefault(detection.track_id, _Track(detection.yolo_name, now, now))
                track.frames += 1
                track.last_seen = now
                area = _area(detection.box)
                if track.label is not None and not 1 / self._max_size_change <= area / max(track.asked_area, 1) <= self._max_size_change:
                    track.label = None
                if self._should_ask(track, detection):
                    track.asked_area = area
                    crop = crop_with_margin(frame, detection.box)
                    track.pending = self._pool.submit(self._describer.describe, crop, detection.yolo_name)
                    track.pending.add_done_callback(lambda future, tid=detection.track_id: self._finish(tid, future))
            for track_id in [t for t, track in self._tracks.items() if now - track.last_seen > self._forget_after_s]:
                del self._tracks[track_id]

    def _priority(self, detection: Detection) -> tuple[int, int]:
        """Old tracks first: on a moving camera half the tracks die within a second, before any answer could arrive."""
        track = self._tracks.get(detection.track_id)
        return (track.frames if track else 0, _area(detection.box))

    def can_name(self, detection: Detection) -> bool:
        x1, y1, x2, y2 = detection.box
        return min(x2 - x1, y2 - y1) >= self._min_side

    def _should_ask(self, track: _Track, detection: Detection) -> bool:
        in_flight = sum(1 for t in self._tracks.values() if t.pending is not None)
        return (
            track.label is None
            and track.pending is None
            and track.failed < self._max_failures
            and track.frames >= self._min_frames
            and self.can_name(detection)
            and in_flight < self._max_pending
        )

    def _finish(self, track_id: int, future: Future) -> None:
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return
            track.pending = None
            try:
                track.label = future.result()
            except Exception as error:  # the next frame retries, up to max_failures
                track.failed += 1
                log.warning("VLM failed on track %s: %s", track_id, error)

    def label(self, track_id: int | None) -> Label | None:
        with self._lock:
            track = self._tracks.get(track_id) if track_id is not None else None
            return track.label if track else None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
