import threading
import time

from PIL import Image

from home_vlm.detect import Detection
from home_vlm.labeler import TrackLabeler, crop_with_margin, mark_box
from home_vlm.vlm import Label


class FakeVLM:
    def __init__(self, delay=0.0, fail=False):
        self.calls = []
        self.delay = delay
        self.fail = fail
        self.release = threading.Event()

    def describe(self, crop, hint):
        self.calls.append((crop.size, hint))
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model down")
        return Label(name=f"真的{hint}")


FRAME = Image.new("RGB", (640, 480))


def det(track_id, box=(100, 100, 200, 220), name="cup"):
    return Detection(track_id=track_id, yolo_name=name, confidence=0.3, box=box)


def wait_for(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_asks_once_per_track_after_it_is_stable():
    vlm = FakeVLM()
    labeler = TrackLabeler(vlm, min_frames=3)
    for i in range(2):
        labeler.update(FRAME, [det(1)], now=i)
    assert vlm.calls == []
    for i in range(2, 10):
        labeler.update(FRAME, [det(1)], now=i)
    assert wait_for(lambda: labeler.label(1) is not None)
    assert len(vlm.calls) == 1
    assert labeler.label(1).name == "真的cup"


def test_track_whose_box_changes_size_is_named_again():
    vlm = FakeVLM()
    labeler = TrackLabeler(vlm, min_frames=1)
    labeler.update(FRAME, [det(1, box=(100, 100, 200, 200))], now=0)
    assert wait_for(lambda: labeler.label(1) is not None)
    labeler.update(FRAME, [det(1, box=(100, 100, 220, 220))], now=1)  # 1.44x: same object
    assert len(vlm.calls) == 1
    labeler.update(FRAME, [det(1, box=(0, 0, 600, 450))], now=2)  # 27x: drifted
    assert wait_for(lambda: len(vlm.calls) == 2)


def test_names_the_biggest_box_first_when_the_model_is_busy():
    vlm = FakeVLM()
    labeler = TrackLabeler(vlm, min_frames=1, max_pending=1)
    labeler.update(FRAME, [det(1, box=(0, 0, 30, 30), name="small"), det(2, box=(0, 0, 300, 300), name="big")], now=0)
    assert wait_for(lambda: vlm.calls)
    assert vlm.calls[0][1] == "big"


def test_skips_tiny_boxes_and_untracked_detections():
    vlm = FakeVLM()
    labeler = TrackLabeler(vlm, min_frames=1, min_side=24)
    labeler.update(FRAME, [det(1, box=(0, 0, 10, 10)), det(None)], now=0)
    time.sleep(0.05)
    assert vlm.calls == []


def test_slow_model_does_not_block_frames_and_caps_in_flight_requests():
    vlm = FakeVLM(delay=0.3)
    labeler = TrackLabeler(vlm, min_frames=1, max_pending=2)
    started = time.perf_counter()
    labeler.update(FRAME, [det(1), det(2), det(3)], now=0)
    assert time.perf_counter() - started < 0.1
    assert wait_for(lambda: labeler.label(2) is not None)
    # The third track was over the cap and gets its turn on a later frame.
    labeler.update(FRAME, [det(3)], now=1)
    assert wait_for(lambda: labeler.label(3) is not None)
    assert len(vlm.calls) == 3


def test_failures_retry_then_give_up():
    vlm = FakeVLM(fail=True)
    labeler = TrackLabeler(vlm, min_frames=1, max_failures=2)
    for i in range(10):
        labeler.update(FRAME, [det(1)], now=i)
        time.sleep(0.02)
    assert labeler.label(1) is None
    assert len(vlm.calls) == 2


def test_forgets_tracks_that_left_the_frame():
    vlm = FakeVLM()
    labeler = TrackLabeler(vlm, min_frames=1, forget_after_s=5)
    labeler.update(FRAME, [det(1)], now=0)
    assert wait_for(lambda: labeler.label(1) is not None)
    labeler.update(FRAME, [], now=6)
    assert labeler.label(1) is None


def test_old_id_after_a_long_gap_does_not_keep_its_name():
    vlm = FakeVLM()
    labeler = TrackLabeler(vlm, min_frames=2, forget_after_s=5)
    labeler.update(FRAME, [det(1)], now=0)
    labeler.update(FRAME, [det(1)], now=0)
    assert wait_for(lambda: labeler.label(1) is not None)
    labeler.update(FRAME, [det(1)], now=6)
    assert labeler.label(1) is None


def test_crop_margin_stays_inside_the_frame():
    crop = crop_with_margin(FRAME, (0, 0, 100, 50))
    assert crop.size == (110, 55)


def test_mark_box_outlines_the_box_inside_its_context():
    view = mark_box(FRAME, (100, 100, 200, 150))
    assert view.size == (300, 150)  # one box-size of context each side
    assert view.getpixel((100, 50)) == (255, 0, 0)  # box corner, shifted into the view
    assert view.getpixel((150, 75)) == (0, 0, 0)  # inside the box is untouched
