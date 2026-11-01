# home-vlm-agent

YOLO finds boxes; a local vision-language model (VLM) names what is in them and answers questions about the scene.

YOLO tracks boxes on every frame. Each track that lasts 15 frames is cropped and sent to the VLM once, oldest track first, with YOLO's guess as a hint. The picture shows the VLM's name; the list beside it and `/api/objects` show both. Chat questions go to the same VLM with the whole frame and the objects named so far.

## Run

```bash
git clone https://github.com/kr-yep/home-vlm-agent.git && cd home-vlm-agent && ./start.sh
```

That is all, on Linux or WSL2 with Docker, Compose 2.30 or later and an NVIDIA GPU that containers can use, about 20 GB of VRAM. The first run builds the images and fetches the weights, which needs the internet and about 50 GB of free disk (29 GB stays afterwards: 13 GB of images and 16 GB of weights; the rest is room for the build, a margin rather than a measurement). Here the first run took 33 minutes with the images already built, nearly all of it the download at about 8 MB/s; building the images from nothing took about 11 minutes more. Later runs only start the app, in about 20 s: run `./start.sh` again. Copying a finished `models/` folder to `$HVA_DATA/models/` beforehand skips the download, since files with the right sha256 are kept. Weights and video are not in this repository.

It opens `http://127.0.0.1:8700`; press 開鏡頭 and allow the camera. Browsers allow a camera only on localhost or HTTPS, so this exact address matters. Keep the tab in front (Chrome slows background tabs) and the camera still; boxes appear at once and names within about 5–15 s.

The model and app containers sit on an `internal` network with no route out; only `proxy` publishes `127.0.0.1:8700`. Stop with `docker compose stop`. After `git pull`, run `docker compose build vlm app`. Tests: `docker run --rm --network none home-vlm-app pytest -q`.

`HVA_SOURCE` picks another input: a video under `/media` (it loops, so it behaves like a camera) or a camera index, for example `HVA_SOURCE=/media/my-room.mp4 docker compose up -d vlm app proxy`. `bench/make_still_clip.py` turns a photo into a clip. AV1 video does not decode in this image; the page then shows the error. A camera index needs the device passed into the container, which is not set up.

Files under `$HVA_DATA` (default `~/.cache/home-vlm`, recorded in `.env`). `start.sh` fetches the first two rows; the rest are only for `bench/`:

| path | source |
|---|---|
| `models/gemma-4-26B_q4_0-it.gguf`, `models/gemma-4-26B-it-mmproj.gguf` | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` @ `d1c082b` (llama.cpp, default) |
| `models/yoloe-26l-o365.pt` | `yoloe-26l-seg.pt` with the Objects365 names built in (`bench/make_prompted_detector.py`, the one step that uses the internet, in a throwaway container with no credentials; default detector) |
| `models/yoloe-26l-seg-pf.pt`, `models/yoloe-26l-lvis.pt` | YOLOE prompt-free, and with the LVIS names built in (comparison only) |
| `models/gemma-4-12b-it-qat-w4a16-ct/` | `google/gemma-4-12B-it-qat-w4a16-ct` @ `1d2c2d7` (vLLM, comparison only) |
| `models/gemma-4-12b-it-qat-q4_0.gguf`, `models/mmproj-…gguf` | `google/gemma-4-12B-it-qat-q4_0-gguf` @ `29d0977` (llama.cpp, Ollama, mistral.rs) |
| `models/gemma-4-12b-it-exl3-4.0bpw/` | `turboderp/gemma-4-12B-it-exl3` @ `0c0ce0e` (ExLlamaV3, comparison only) |
| `models/sglang-config.json` | the w4a16 `config.json` plus `audio_config.output_proj_dims = 640` (SGLang, comparison only) |
| `models/yolo.pt` → `yolo26m.pt` | ultralytics/assets v8.4.0, sha256 `401cea9a…d0b7` |
| `media/` | test clips; sources and licences in `media/SOURCES.txt` |
| `data/lvis-eval/` | 500 objects from LVIS v1 val (`python -m bench.prepare_lvis`) |
| `runs/` | benchmark and evaluation outputs |

## Status (2026-11-01)

YOLO → VLM works end to end in the real app and page (default stack, office clip, browser): all 10 boxes named, chat answers in about 0.8 s, 19.1 GB VRAM peak with YOLO. Neither the app nor the VLM container can reach the internet.

Live camera: `start.sh` from a clone of this repository and an empty data folder, then the page in Chromium with a fake camera device playing a still photo: 15 frames/s reached the app, first name at 6.2 s, all 13 by 13.3 s (a second scene: 3.0 s, all 19 by 13.7 s), chat about 1 s, camera off and on again worked. Run on WSL2 with an RTX 3090 and Docker already installed. No physical camera, RTX 5090, plain Linux or fresh machine has been tried yet.

It is good with a still camera and poor with a moving one. On still clips every box above the size limit had a name by the end of a 45 s run (median 7–17 s after it first appeared). On handheld or panning clips few boxes on screen have a name, because half the tracks die within about a second, before an answer can arrive. Not done: memory of objects across time, multiple cameras.

Known weak spots: if the camera turns to a new scene while it is on, a new object overlapping an old box can keep the old name until the tracker drops it (turning the camera off for more than 10 s clears this); the VLM is sometimes confidently wrong (microwave → 3D 列印機, pillow → 麵包); questions that need reasoning are unreliable ("what could I cook with?" answered "nothing" beside a fridge); now and then a chat answer repeats until the token limit.

### Which detector and VLM

500 indoor LVIS objects, one run each. End to end: the detector boxed the object (IoU ≥ 0.5) and the name is right (WordNet lenient), with missed objects counted as wrong. `bench/recognition_eval.py`, `bench/score.py`, `bench/cross.py`; boxes alone in `bench/proposers.py`. Both VLMs on llama.cpp b11869. YOLOE-26l is used throughout.

| detector, threshold 0.2 | boxed | boxes per image | detector's own name | + Gemma 12B | + Gemma 26B |
|---|---|---|---|---|---|
| YOLO26m (COCO 80) | 44.4% | 9.5 | 35.2% | 27.6% | 29.6% |
| YOLOE, Objects365 names | 72.4% | 22.5 | 40.4% | 40.2% | 43.2% |
| YOLOE, prompt-free (4585 tags) | 84.4% | 32.1 | 41.2% | 46.0% | 49.0% |
| YOLOE, LVIS names | 81.2% | 23.5 | 36.6% | not run | 50.6% |

- The detector matters most. YOLO26m against any YOLOE row: p < 1e-8 (paired sign test). Its weakness is outside COCO: 4–5% end to end on the 250 other objects, against 32–46%. On COCO classes its own name, 69%, is still the best number here, measured with a favourable hand-made name map.
- 26B beats 12B on the true box (53.6% vs 45.2%, p = 6.6e-5). End to end the gap is 2–3 points and not significant (p = 0.10–0.15), and it is faster (median 0.55 s vs 0.9–1.0 s).
- Prompt-free against Objects365 with 26B: p = 0.0006. Prompt-free against LVIS names: p = 0.28. The LVIS row is flattered because its vocabulary is the test's.
- These do not measure wrong boxes: LVIS labels are incomplete, so a box with no label is not necessarily wrong. That only shows on real footage, below.
- Raising the threshold (read from the same run, approximately): Objects365 + 26B gives 39.8% at 0.3 and 34.8% at 0.5; prompt-free gives 47.6% and 42.4%.

Real clips through the whole app with 26B, 45 s, one run each, boxes named / boxes on screen in the last 10 s:

| detector, threshold | office | bedroom | kitchen | panning bedroom | handheld kitchen |
|---|---|---|---|---|---|
| **YOLOE Objects365, 0.3** | 10 / 10 | 18 / 18 | 17.9 / 17.9 | 1.6 / 10.2 | 0.2 / 3.6 |
| YOLOE Objects365, 0.5 | 8.9 / 8.9 | 17.9 / 17.9 | 14 / 14 | 0.5 / 7.5 | 0 / 1.7 |
| YOLOE LVIS names, 0.3 | 7.9 / 8 | 19.8 / 19.8 | 15.8 / 15.8 | 1.8 / 12.2 | 0.3 / 5.2 |
| YOLOE prompt-free, 0.3 | 13 / 13.1 | 22.9 / 22.9 | 18.9 / 18.9 | 1.6 / 15.9 | 0.9 / 13.1 |
| YOLOE prompt-free, 0.5 | 10 / 10 | 20.8 / 20.8 | 15.6 / 15.6 | 2.6 / 11.9 | 2.0 / 6.4 |
| YOLO26m, 0.2 | 8.9 / 8.9 | 4 / 4 | 6.8 / 6.8 | 0.9 / 3.6 | 0.3 / 1.7 |

The still clips are COCO photos looped, standing in for a camera on a tripod, and they are also in the 500-object test.

## Decisions

- **Stack: YOLOE-26l with Objects365 names at 0.3, BoT-SORT, Gemma 4 26B on llama.cpp.** Prompt-free scores higher on LVIS, but on real footage it draws large boxes over walls and ceilings and proposes scene words ("veterinarians office", "biology laboratory"). The VLM then names those boxes (a wall as 電腦螢幕), which also misleads chat. Objects365 keeps to objects; the cost is real and measured, about 8 points at this threshold. To switch: `HVA_YOLO=/models/yoloe-26l-seg-pf.pt HVA_CONF=0.5`. Compared by eye on saved frames, not by a score. Thresholds were chosen on these same clips, so the table above is not an independent test.
- **Tracker: named explicitly.** Ultralytics 8.4's default, TrackTrack, starts a track only at 0.7 confidence whatever `conf` says (kitchen clip: 16 boxes against 35 with BoT-SORT at the same 0.2; YOLO26m 3 against 14).
- **Runtime: llama.cpp for the 26B.** No w4a16 weights exist for it, so vLLM cannot run it. At 15 fps with YOLO: label 0.87 s median, chat 0.42 s, against 1.07 s and 0.39 s for the 12B on vLLM; 19.1 GB against 14.3 GB peak VRAM. That was measured before the tracker fix.
- **12B runtimes, kept for reference.** vLLM 0.32.0 won this comparison and was the default until the 26B replaced it. Same 30 crops, YOLO running, one runtime on the GPU at a time, fresh server each (`bench/runtime_bench.py`); recognition on the 500 objects above (`bench/recognition_eval.py`). VRAM includes YOLO and about 1.1 GB the Windows desktop holds.

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

  Accuracy on the app's path could not be told apart between runtimes, and vLLM was among the fastest at both labelling and chat while labelling. The gaps are small (0.1–0.3 s per label); llama.cpp's lead on the app's path (69.8% vs 65.8%) is not significant but would be worth re-measuring on more objects. SGLang ties vLLM on speed, but in this setup, on the same weights, it names true-box crops much worse without a hint ("handheld microphone" for a wine glass). The cause is not known; its Triton attention backend on this GPU and the patched `config.json` were not examined, so this does not show SGLang's vision is worse in general. Peak VRAM follows each runtime's reservation setting (vLLM 0.5 of the GPU, mistral.rs 8192 KV tokens, ExLlamaV3 a 32768-token cache) rather than what it needs. Startup, measured once: SGLang 225 s (mostly CUDA graph capture), mistral.rs 16 s, ExLlamaV3 15 s.

  Not run: KoboldCpp (llama.cpp-based). Ruled out from their docs and source, not by running them: TensorRT-LLM (cannot load these 4-bit weights), LMDeploy, TGI and MLC-LLM (no Gemma 4 12B vision), ik_llama.cpp (no vision for the 12B). None of the runtimes was tuned beyond what it took to serve; each column is one run.
- **When the VLM runs.** Once per track, after 15 frames, oldest track first, on 3 background workers; the VLM has 4 slots, so one stays free for chat. A track's name is dropped and asked again if its box changes size more than 4×, since it has probably drifted. Frames never wait for the model. Going from 3 frames and 2× to 15 and 4× named more on the panning clip (24 tracks against 9) and about the same on the rest, and 1 worker against 3 made little difference; one run each, both measured before the tracker fix.
- **Boxes under 24 pixels a side** are not shown or named: the crop is too small to name, and they would stay "unnamed" for ever.
- **Hint.** The detector's guess is passed with "it is often wrong". End to end it helps: 43.2% against 39.6% with Objects365 + 26B (p = 0.011).
- **Supply chain.** Python deps resolve only releases at least 7 days old (`exclude-newer = "7 days"`) and are hash-locked in `uv.lock`. llama.cpp and the TabbyAPI source come from checksum-pinned tarballs; TabbyAPI's Python deps (comparison only) are bounded by date, not hash-locked, and torch and exllamav3 are direct URLs without hashes. Images are pinned by tag or digest and were built at least 7 days ago. Weights are checked by sha256.
- **Camera through the browser.** The page captures frames and posts them to `/api/frame`, sending the next only after the last returns, and the app keeps only the newest. So no camera device has to pass into Docker, and a slow detector drops frames rather than falling behind. Frames travel only to `127.0.0.1`.
- **Licence.** This repository is AGPL-3.0 because it uses Ultralytics YOLO, which is. The weights are not included and have their own terms.
