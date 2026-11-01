# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.13.5-python3.14-trixie-slim

# libGL/glib for OpenCV, Noto CJK so overlays can draw Chinese labels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    YOLO_OFFLINE=True \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY README.md ./
COPY src ./src
COPY tests ./tests
COPY bench ./bench
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync"]
CMD ["python", "-m", "home_vlm"]
