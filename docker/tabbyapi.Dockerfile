# TabbyAPI + ExLlamaV3 1.6.4 (torch 2.9.0+cu128), for the runtime comparison only.
# TabbyAPI commit 9979d01 (2026-10-21) is the last one pinning exllamav3 1.6.4 (2026-10-20);
# the source tarball is checksum-checked and PyPI deps resolve only releases up to 2026-10-25.
FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:0.13.5-python3.14-trixie-slim@sha256:d8e732839dfc4225a1ff698803b478b2223c05636c055d3add41ab9dbec0f315 /usr/local/bin/uv /usr/local/bin/uv
# Triton compiles a small extension at runtime, so it needs gcc and Python.h.
RUN sed -i 's#http://\(archive\|security\).ubuntu.com#http://tw.archive.ubuntu.com#' /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-dev build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
ARG TABBY=9979d0188c3e687a6ba9ed3ddb7db79048116116
ARG TABBY_SHA256=889c9210eeebae7bb469396212460bc4d7c04f15241d25d52c4bf7b903260322
WORKDIR /app
RUN curl -fsSL https://codeload.github.com/theroyallab/tabbyAPI/tar.gz/$TABBY -o /tmp/tabby.tgz \
 && echo "$TABBY_SHA256  /tmp/tabby.tgz" | sha256sum -c - \
 && tar -xzf /tmp/tabby.tgz --strip-components=1 && rm /tmp/tabby.tgz
ENV UV_HTTP_TIMEOUT=300 PATH=/opt/venv/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/bin/python3.12 \
 && uv pip install --python /opt/venv/bin/python --exclude-newer 2026-10-25T00:00:00Z ".[cu12]"
EXPOSE 5000
# Triton links with -lcuda, but the GPU device mounts only libcuda.so.1, under a
# driver-specific path on WSL, so the unversioned link is made at startup.
ENTRYPOINT ["sh", "-c", "ln -sf \"$(ldconfig -p | awk '/libcuda.so.1/{print $NF; exit}')\" /usr/lib/x86_64-linux-gnu/libcuda.so; exec python main.py \"$@\"", "--"]
