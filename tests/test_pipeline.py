from home_vlm import pipeline as pipeline_module
from home_vlm.pipeline import Pipeline


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
