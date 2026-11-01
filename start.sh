#!/usr/bin/env bash
# The one command. The first run prepares the machine (needs Docker, an NVIDIA GPU that
# containers can use, about 50 GB of free disk and the internet); after that it only
# starts the app. Safe to interrupt and run again: finished downloads are kept.
set -euo pipefail
cd "$(dirname "$0")"

saved=$(sed -n 's/^HVA_DATA=//p' .env 2>/dev/null || true)
export HVA_DATA="${HVA_DATA:-${saved:-$HOME/.cache/home-vlm}}"
mkdir -p "$HVA_DATA/models" "$HVA_DATA/media" "$HVA_DATA/runs"
echo "HVA_DATA=$HVA_DATA" > .env

if [ ! -f "$HVA_DATA/models/.ready" ]; then
  docker compose build vlm app

  # Before the long download: a machine that cannot use the GPU stops here.
  docker run --rm --gpus all --entrypoint /app/.venv/bin/python home-vlm-app -c '
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
torch.ones(1, device="cuda").sum()
print("GPU ok:", torch.cuda.get_device_name(0))'
  docker run --rm --gpus all --entrypoint llama-server home-vlm-llamacpp:b11869 --list-devices 2>&1 | grep "CUDA0:"

  GEMMA=https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/d1c082be9cf3c8a514acf63b8761f4b41935842e
  ASSETS=https://github.com/ultralytics/assets/releases/download/v8.4.0
  docker run --rm -i --entrypoint bash -v "$HVA_DATA/models:/models" home-vlm-llamacpp:b11869 -s <<SCRIPT
set -euo pipefail
cd /models
fetch() {  # file url sha256
  if [ -f "\$1" ] && echo "\$3  \$1" | sha256sum -c --status; then echo "ok \$1"; return; fi
  curl -fL -C - --retry 20 --retry-all-errors -o "\$1" "\$2"
  echo "\$3  \$1" | sha256sum -c
}
fetch gemma-4-26B_q4_0-it.gguf $GEMMA/gemma-4-26B_q4_0-it.gguf 3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d
fetch gemma-4-26B-it-mmproj.gguf $GEMMA/gemma-4-26B-it-mmproj.gguf a359953a076b877db30c31dbbb4c6d93b4a6e017ee5db5784247e4d4c0dd4f3b
fetch yoloe-26l-seg.pt $ASSETS/yoloe-26l-seg.pt a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9
fetch mobileclip2_b.ts $ASSETS/mobileclip2_b.ts 35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f
SCRIPT

  if [ ! -f "$HVA_DATA/models/yoloe-26l-o365.pt" ]; then
    # The one step that needs the internet inside the app image: a throwaway container, no credentials.
    docker run --rm --entrypoint sh -e YOLO_OFFLINE=false -e CUDA_VISIBLE_DEVICES= \
      -v "$HVA_DATA/models:/models" -v "$PWD/bench:/tmp/bench:ro" home-vlm-app -c '
      uv pip install -q --python /app/.venv/bin/python --exclude-newer 2026-10-25T00:00:00Z \
        https://github.com/ultralytics/CLIP/archive/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40.tar.gz
      /app/.venv/bin/python /tmp/bench/make_prompted_detector.py yoloe-26l-seg.pt Objects365 yoloe-26l-o365.pt'
  fi
  touch "$HVA_DATA/models/.ready"
fi

docker compose up -d --wait vlm app proxy

URL=http://127.0.0.1:8700
echo "Ready: open $URL in a browser on this machine and press 開鏡頭."
if command -v explorer.exe >/dev/null; then
  explorer.exe "$URL" || true  # always exits 1
elif command -v xdg-open >/dev/null; then
  xdg-open "$URL" >/dev/null 2>&1 || true
fi
