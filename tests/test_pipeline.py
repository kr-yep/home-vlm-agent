import io

import numpy as np
from PIL import Image

from home_vlm import pipeline as pipeline_module
from home_vlm.pipeline import BrowserCamera, Pipeline


class Unreadable:
    """A video that opens but never yields a frame, like AV1 without a decoder."""

    def get(self, _prop):
        return 30.0

    def read(self):
        return False, None

    def set(self, *_args):
        return True

    def release(self):
        pass


def test_video_that_opens_but_never_decodes_is_reported(monkeypatch):
    monkeypatch.setattr(pipeline_module, "open_source", lambda source: Unreadable())
    monkeypatch.setattr(pipeline_module, "FIRST_FRAME_TIMEOUT_S", 0.0)
    loop = Pipeline("clip.webm", detector=None, labeler=None)
    loop.run()  # returns instead of looping forever
    assert "clip.webm" in loop.error



def test_what_the_browser_last_showed_expires_when_it_stops_sending(monkeypatch):
    class NoBoxes:
        def track(self, _frame):
            return []

    class NoNames:
        def update(self, _frame, _detections):
            pass

    loop = Pipeline(BrowserCamera(), NoBoxes(), NoNames())
    assert loop.snapshot() == (None, [], None)
    loop.step(np.zeros((8, 8, 3), np.uint8))
    assert loop.snapshot()[0] is not None
    monkeypatch.setattr(pipeline_module, "BROWSER_IDLE_S", -1.0)
    assert loop.snapshot() == (None, [], None)


def test_browser_camera_keeps_only_the_newest_frame():
    camera = BrowserCamera()
    for colour in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), colour).save(buffer, format="PNG")
        assert camera.put(buffer.getvalue())
    _, frame = camera.read()
    assert frame[0, 0, 0] == 255  # OpenCV is BGR, so blue
