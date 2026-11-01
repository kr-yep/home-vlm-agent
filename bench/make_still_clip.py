"""Turn a photo into a 10 s still clip, a stand-in for a camera on a tripod.

    docker run --rm --network none --entrypoint /app/.venv/bin/python \
        -v /root/.cache/home-vlm/media:/media -v $PWD/bench:/tmp/bench:ro home-vlm-app \
        /tmp/bench/make_still_clip.py /media/my-room.jpg /media/my-room.mp4
"""

from __future__ import annotations

import sys

import cv2

FPS = 15
SECONDS = 10


def main() -> None:
    photo, out = sys.argv[1:3]
    image = cv2.imread(photo)
    if image is None:
        raise SystemExit(f"cannot read {photo}")
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    for _ in range(FPS * SECONDS):
        writer.write(image)
    writer.release()
    print("wrote", out, width, height)


if __name__ == "__main__":
    main()
