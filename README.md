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
| `models/gemma-4-12b-it-qat-q4_0.gguf`, `models/mmproj-…gguf` | `google/gemma-4-12B-it-qat-q4_0-gguf` @ `29d0977` (llama.cpp, Ollama, mistral.rs) |
| `models/gemma-4-12b-it-exl3-4.0bpw/` | `turboderp/gemma-4-12B-it-exl3` @ `0c0ce0e` (ExLlamaV3, comparison only) |
| `models/sglang-config.json` | the w4a16 `config.json` plus `audio_config.output_proj_dims = 640` (SGLang, comparison only) |
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

- **Runtime: vLLM 0.32.0** with Google's w4a16 QAT weights. Same 30 crops, YOLO running, one runtime on the GPU at a time, fresh server each (`bench/runtime_bench.py`); recognition on the 500 objects above (`bench/recognition_eval.py`). VRAM includes YOLO and about 1.1 GB the Windows desktop holds.

  | | vLLM 0.32.0 | SGLang 0.5.22 | mistral.rs 0.9.5 | ExLlamaV3 1.6.4 (TabbyAPI) | Ollama 0.40.10 | llama.cpp b11869 |
  |---|---|---|---|---|---|---|
  | weights | w4a16 | w4a16 | GGUF q4_0 | EXL3 4.0 bpw | GGUF q4_0 | GGUF q4_0 |
  | label, median / p95 | 0.84 / 0.92 s | 0.83 / 0.89 s | 1.02 / 1.12 s | 1.05 / 1.55 s | 0.91 / 1.09 s | 1.15 / 1.27 s |
  | first token | 0.15 s | 0.16 s | 0.40 s | 0.35 s | 0.25 s | 0.34 s |
  | decode | 66 tok/s | 69 tok/s | 70 tok/s | 66 tok/s | 67 tok/s | 74 tok/s |
  | chat while labelling, median | 0.35 s | 0.32 s | 0.64 s | 0.75 s | 1.44 s (0.75 s at 4 slots) | 1.42 s (0.54 s at 4 slots) |
  | peak VRAM | 13.8 GB | 15.0 GB | 12.0 GB | 11.1 GB | 10.0 GB (12.3 GB) | 9.7 GB (11.5 GB) |
  | named right on the true box (lenient) | 49.2% | 34.2% | 48.2% | 48.0% | not run | 45.2% |
  | named right on YOLO's box, with hint (COCO-level) | 65.8% | 62.6% | 64.0% | 63.5% | not run | 69.8% |

  vLLM, SGLang, mistral.rs and ExLlamaV3 ran with room for 4 concurrent requests; Ollama and llama.cpp with 1 unless noted. Paired sign tests against vLLM. True box (500 objects): SGLang p ≈ 5e-12, llama.cpp p = 0.02, mistral.rs p = 0.6, ExLlamaV3 p = 0.55. YOLO's box with hint, the path the app uses (222 objects): llama.cpp p = 0.06, SGLang p = 0.35, mistral.rs p = 0.50, ExLlamaV3 p = 0.42, so none is distinguishable from vLLM there.

  vLLM stays the default: accuracy on the app's path cannot be told apart between runtimes, and vLLM is among the fastest at both labelling and chat while labelling. The gaps are small (0.1–0.3 s per label); llama.cpp's lead on the app's path (69.8% vs 65.8%) is not significant but would be worth re-measuring on more objects. SGLang ties vLLM on speed, but in this setup, on the same weights, it names true-box crops much worse without a hint ("handheld microphone" for a wine glass). The cause is not known; its Triton attention backend on this GPU and the patched `config.json` were not examined, so this does not show SGLang's vision is worse in general. Peak VRAM follows each runtime's reservation setting (vLLM 0.5 of the GPU, mistral.rs 8192 KV tokens, ExLlamaV3 a 32768-token cache) rather than what it needs. Startup, measured once: SGLang 225 s (mostly CUDA graph capture), mistral.rs 16 s, ExLlamaV3 15 s.

  Not run: KoboldCpp (llama.cpp-based). Ruled out from their docs and source, not by running them: TensorRT-LLM (cannot load these 4-bit weights), LMDeploy, TGI and MLC-LLM (no Gemma 4 12B vision), ik_llama.cpp (no vision for the 12B). None of the runtimes was tuned beyond what it took to serve; each column is one run.
- **When the VLM runs.** Once per track, after 3 frames, on one background worker (one request running, one queued). Frames never wait for the model.
- **Hint.** YOLO's guess is passed with "it is often wrong". With the hint the VLM does better on COCO classes (66% vs 58% lenient on YOLO boxes) and slightly worse on other objects, where it copies YOLO's wrong class about a third of the time.
- **Supply chain.** Python deps resolve only releases at least 7 days old (`exclude-newer = "7 days"`) and are hash-locked in `uv.lock`. llama.cpp and the TabbyAPI source come from checksum-pinned tarballs; TabbyAPI's Python deps (comparison only) are bounded by date, not hash-locked, and torch and exllamav3 are direct URLs without hashes. Images are pinned by tag or digest and were built at least 7 days ago. Weights are checked by sha256.
- **Licence.** Ultralytics YOLO is AGPL-3.0. Fine for coursework; anything distributed or served to others must follow AGPL or use a differently licensed detector.
