# home-vlm-agent

YOLO finds boxes; a local vision-language model (VLM) names what is in them and answers questions about the scene.

YOLO tracks boxes on every frame (low confidence threshold). Each track that lasts 3 frames is cropped and sent to the VLM once, with YOLO's guess as a hint. The page and `/api/objects` show both names. Chat questions go to the same VLM with the current object list and the largest crop.

## Run

Everything runs in Docker inside WSL (Ubuntu-24.04, NVIDIA CDI). The model and app containers sit on an `internal` network with no route out; only `proxy` publishes `127.0.0.1:8700`.

```bash
# in WSL, from this folder
docker compose up -d vlm app proxy        # open http://127.0.0.1:8700
docker run --rm --network none home-vlm-app pytest -q
```

`HVA_SOURCE` picks the input: a camera index (`0`) or a video under `/media` (default `kitchen-hotel.webm`). Camera passthrough into Docker on WSL is not set up or tested.

Files in `/root/.cache/home-vlm` (WSL), not in Git:

| path | source |
|---|---|
| `models/gemma-4-12b-it-qat-w4a16-ct/` | `google/gemma-4-12B-it-qat-w4a16-ct` @ `1d2c2d7` (vLLM, default) |
| `models/gemma-4-12b-it-qat-q4_0.gguf`, `models/mmproj-…gguf` | `google/gemma-4-12B-it-qat-q4_0-gguf` @ `29d0977` (llama.cpp, Ollama) |
| `models/yolo.pt` → `yolo26m.pt` | ultralytics/assets v8.4.0, sha256 `401cea9a…d0b7` |
| `media/` | Wikimedia Commons clips; licences in `media/SOURCES.txt` |
| `data/lvis-eval/` | 500 objects from LVIS v1 val (`python -m bench.prepare_lvis`) |
| `runs/` | benchmark and evaluation outputs |

## Status (2026-11-01)

Works end to end on a recorded kitchen video: YOLO tracks, the VLM names tracks in about 1.5 s (median, first seen → named), chat answers in 0.2–0.8 s, 13.9 GB VRAM peak with YOLO. The containers resolve no outside hosts. Not done: live camera, memory of objects across time, multiple cameras.

Recognition on 500 indoor LVIS objects (`bench/recognition_eval.py`, `bench/score.py`, `bench/fusion.py`; Gemma 4 12B QAT on llama.cpp unless noted):

| | COCO classes (250) | other household objects (250) |
|---|---|---|
| YOLO finds a box (IoU ≥ 0.5) | 74% | 15% |
| YOLO's name right, on its boxes | 93.5% | 8% |
| VLM's name right, on YOLO's boxes, with hint (COCO-level yardstick) | 77% (vLLM 73%) | 35% (vLLM 30%) |
| VLM's name right, given the true box (WordNet lenient) | 49% (vLLM 55%) | 42% (vLLM 44%) |

A hand check of 80 automatic misses found 16 where the answer was in fact right, 17 where the VLM named another object in the crop (the cat on the bed), 37 plain misrecognitions and 10 unclear; the corrected rates are roughly 5–8 points higher. Showing the box in red on a wider view did not help (44.6% vs 45.2% on true boxes).

What this means: on the 80 COCO classes YOLO's own name is already better than the VLM's. The VLM adds names for things outside COCO and descriptions (colour, material, what is in the pan), but YOLO only boxes 15% of those objects, so the proposer, not the namer, is the limit.

## Decisions

- **Runtime: vLLM 0.32.0** with Google's w4a16 QAT weights. Same 30 crops, YOLO running, one runtime on the GPU at a time (`bench/runtime_bench.py`, fresh server each):

  | | vLLM | Ollama 0.40.10 | llama.cpp b11869 |
  |---|---|---|---|
  | label, median / p95 | 0.84 / 0.92 s | 0.91 / 1.09 s | 1.15 / 1.27 s |
  | first token | 0.15 s | 0.25 s | 0.34 s |
  | decode | 66 tok/s | 67 tok/s | 74 tok/s |
  | chat while labelling, median | 0.35 s | 1.44 s | 1.42 s |
  | peak VRAM incl. YOLO | 13.8 GB | 10.0 GB | 9.7 GB |

  Ollama and llama.cpp load the same GGUF; vLLM loads the w4a16 sibling and sees about 395 image tokens per crop instead of 212, with equal or better accuracy. llama.cpp writes longer JSON (60 tokens vs 43–47), which is most of its gap. vLLM batches up to 4 requests; llama.cpp and Ollama ran one at a time.
- **When the VLM runs.** Once per track, after 3 frames, on one background worker (one request running, one queued). Frames never wait for the model.
- **Hint.** YOLO's guess is passed with "it is often wrong". With the hint the VLM does better on COCO classes (66% vs 58% lenient on YOLO boxes) and slightly worse on other objects, where it copies YOLO's wrong class about a third of the time.
- **Supply chain.** Python deps resolve only releases at least 7 days old (`exclude-newer = "7 days"`) and are hash-locked in `uv.lock`. llama.cpp comes from a checksum-pinned release tarball; images are pinned by tag and were built at least 7 days ago. Weights are checked by sha256.
- **Licence.** Ultralytics YOLO is AGPL-3.0. Fine for coursework; anything distributed or served to others must follow AGPL or use a differently licensed detector.
