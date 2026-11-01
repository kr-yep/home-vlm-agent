"""Save YOLOE with a dataset's class names built in, so the app can run it offline.

This one step needs the internet (Ultralytics CLIP and its text encoder), so it runs
in a throwaway container with no credentials, never on the app's network. Run from WSL:
    docker run --rm --entrypoint sh -e YOLO_OFFLINE=false -e CUDA_VISIBLE_DEVICES= \
        -v /root/.cache/home-vlm/models:/models -v $PWD/bench:/tmp/bench:ro home-vlm-app -c '
        uv pip install -q --python /app/.venv/bin/python --exclude-newer 2026-10-25T00:00:00Z \
            https://github.com/ultralytics/CLIP/archive/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40.tar.gz
        /app/.venv/bin/python /tmp/bench/make_prompted_detector.py yoloe-26l-seg.pt Objects365 yoloe-26l-o365.pt'
"""

from __future__ import annotations

import os
import pathlib
import sys

import ultralytics
import yaml
from ultralytics import YOLOE


def main() -> None:
    weights, dataset, out = sys.argv[1:4]
    config = pathlib.Path(ultralytics.__file__).parent / f"cfg/datasets/{dataset}.yaml"
    names = [name.lower() for name in yaml.safe_load(config.read_text())["names"].values()]
    os.chdir("/models")
    model = YOLOE(weights)
    model.set_classes(names, model.get_text_pe(names))
    model.save(out)
    print("saved", out, "with", len(names), "classes")


if __name__ == "__main__":
    main()
