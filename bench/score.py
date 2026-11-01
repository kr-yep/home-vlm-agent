"""Score recognition.jsonl against LVIS categories.

  strict:  the English name, minus leading adjectives, is one of the LVIS
           category's synonyms (for YOLO: a hand-mapped class).
  lenient: strict, or the name contains a synonym ("toilet paper roll"), or
           WordNet agrees: the same synset, a more specific one, or the direct
           parent when that parent is narrow (<= 30 descendants), so "container"
           or "object" never count.
Run with the bench group (needs WordNet; first run downloads it to /data/nltk_data):
    docker run --rm -v /root/.cache/home-vlm:/data -v $PWD:/app -w /app <uv image> \
        uv run --frozen --group bench python -m bench.score /data/runs/recognition.jsonl
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import nltk

NLTK_DATA = os.environ.get("NLTK_DATA", "/data/nltk_data")
nltk.data.path.insert(0, NLTK_DATA)
try:
    from nltk.corpus import wordnet as wn

    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet", download_dir=NLTK_DATA, quiet=True)
    from nltk.corpus import wordnet as wn

from .coco_lvis import COCO_TO_LVIS, yolo_correct

NARROW_PARENT = 30
ARTICLES = ("a ", "an ", "the ")
CUT_WORDS = {"of", "with", "on", "in", "for", "from", "under", "near", "next", "inside"}


def normalise(name: str) -> str:
    # "built-in" stays one word (so "in" is not read as a preposition); WordNet
    # spells hyphenated nouns with "-", which `lenient` tries as well.
    name = name.strip().lower().replace("-", "_")
    for article in ARTICLES:
        if name.startswith(article):
            name = name[len(article) :]
    return "_".join(name.split())


def core(name: str) -> str:
    """Drop what follows " of / with / on / in ...": "cat on a bed" names a cat."""

    words = normalise(name).split("_")
    for i, word in enumerate(words):
        if i and word in CUT_WORDS:
            return "_".join(words[:i])
    return "_".join(words)


def suffixes(phrase: str) -> list[str]:
    """"decorated_christmas_tree" -> itself, "christmas_tree", "tree" (longest first)."""

    words = phrase.split("_")
    return ["_".join(words[i:]) for i in range(len(words))]


def strict(name: str, obj: dict) -> bool:
    """The answer, minus adjectives, is one of the LVIS synonyms."""

    synonyms = {normalise(s) for s in obj["synonyms"]} | {normalise(obj["category"])}
    for phrase in suffixes(core(name)):
        if phrase in synonyms or (wn.morphy(phrase, wn.NOUN) or phrase) in synonyms:
            return True
    return False


def contains_synonym(name: str, obj: dict) -> bool:
    """"pink toilet paper roll" contains the synonym "toilet paper"."""

    phrase = "_" + core(name) + "_"
    return any("_" + normalise(s) + "_" in phrase for s in obj["synonyms"])


def lenient(name: str, obj: dict) -> bool:
    if strict(name, obj) or contains_synonym(name, obj):
        return True
    try:
        target = wn.synset(obj["synset"])
    except Exception:
        return False
    parents = set(target.hypernyms()) | set(target.instance_hypernyms())
    # The longest tail WordNet knows: "decorated christmas tree" -> christmas_tree, not tree.
    senses = []
    for phrase in suffixes(core(name)):
        senses = [s for form in (phrase, phrase.replace("_", "-"), phrase.replace("_", "")) for s in wn.synsets(form, pos=wn.NOUN)]
        if senses:
            break
    for sense in senses:
        if sense == target or target in sense.closure(lambda s: s.hypernyms() + s.instance_hypernyms()):
            return True
        if sense in parents and len(list(sense.closure(lambda s: s.hyponyms()))) <= NARROW_PARENT:
            return True
    return False


def coarse(name: str, obj: dict) -> bool:
    """Judged the way YOLO is: the answer falls in the same COCO-class group as the
    object ("mug" for a cup counts, as YOLO's "cup" for a mug does)."""

    for coco, group in COCO_TO_LVIS.items():
        if obj["category"] in group:
            names = {coco} | {g.split("_(")[0] for g in group}
            return any(strict(name, {"synonyms": [n], "category": n}) for n in names)
    return False


def rate(values) -> str:
    values = list(values)
    return f"{sum(values) / len(values):.1%} ({sum(values)}/{len(values)})" if values else "n/a"


def main() -> None:
    path = Path(sys.argv[1])
    manifest = {o["id"]: o for o in json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "/data/data/lvis-eval/manifest.json").read_text())}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for r in records:
        obj = manifest[r["id"]]
        row = {"id": r["id"], "category": r["category"], "split": r["split"], "yolo_class": r["yolo_class"]}
        row["yolo_matched"] = r["yolo_class"] is not None
        row["yolo_strict"] = yolo_correct(r["yolo_class"], r["category"])
        row["yolo_lenient"] = row["yolo_strict"] or (r["yolo_class"] is not None and lenient(r["yolo_class"], obj))
        for key in ("gt_nohint", "yolo_hint", "yolo_nohint"):
            answer = r.get(key)
            if answer is None:
                continue
            name = answer.get("name_en", "")
            row[key] = {
                "name_en": name,
                "name": answer.get("name", ""),
                "error": "error" in answer,
                "parsed": answer.get("parsed", False),
                "strict": bool(name) and strict(name, obj),
                "lenient": bool(name) and lenient(name, obj),
                "coarse": bool(name) and (lenient(name, obj) or coarse(name, obj)),
                "latency_s": answer.get("latency_s"),
                "copied": r["yolo_class"] is not None and strict(name, {"synonyms": [r["yolo_class"]], "category": r["yolo_class"]}),
            }
        rows.append(row)

    report: dict[str, dict] = {}
    for split in ("coco", "non_coco", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        matched = [r for r in sel if r["yolo_matched"]]
        yolo_wrong = [r for r in matched if not r["yolo_strict"] and "yolo_hint" in r]
        section = {
            "objects": len(sel),
            "yolo_box_found (IoU>=0.5)": rate(r["yolo_matched"] for r in sel),
            "YOLO name correct, strict (all objects)": rate(r["yolo_strict"] for r in sel),
            "YOLO name correct, lenient (all objects)": rate(r["yolo_lenient"] for r in sel),
            "VLM on GT box, strict": rate(r["gt_nohint"]["strict"] for r in sel if "gt_nohint" in r),
            "VLM on GT box, lenient": rate(r["gt_nohint"]["lenient"] for r in sel if "gt_nohint" in r),
            "YOLO+VLM app path, lenient (all objects)": rate(r.get("yolo_hint", {}).get("lenient", False) for r in sel),
            "on YOLO-found boxes: YOLO strict": rate(r["yolo_strict"] for r in matched),
            "on YOLO-found boxes: VLM with hint, strict": rate(r["yolo_hint"]["strict"] for r in matched if "yolo_hint" in r),
            "on YOLO-found boxes: VLM with hint, lenient": rate(r["yolo_hint"]["lenient"] for r in matched if "yolo_hint" in r),
            "on YOLO-found boxes: VLM with hint, coarse (YOLO's yardstick)": rate(r["yolo_hint"]["coarse"] for r in matched if "yolo_hint" in r),
            "VLM on GT box, coarse (YOLO's yardstick)": rate(r["gt_nohint"]["coarse"] for r in sel if "gt_nohint" in r),
            "on YOLO-found boxes: VLM no hint, lenient": rate(r["yolo_nohint"]["lenient"] for r in matched if "yolo_nohint" in r),
            "YOLO wrong: VLM with hint fixed it": rate(r["yolo_hint"]["lenient"] for r in yolo_wrong),
            "YOLO wrong: VLM with hint copied YOLO": rate(r["yolo_hint"]["copied"] for r in yolo_wrong),
            "YOLO wrong: VLM no hint fixed it": rate(r["yolo_nohint"]["lenient"] for r in yolo_wrong),
            "YOLO wrong: VLM no hint said YOLO's class": rate(r["yolo_nohint"]["copied"] for r in yolo_wrong),
        }
        report[split] = section

    answers = [r[k] for r in rows for k in ("gt_nohint", "yolo_hint", "yolo_nohint") if k in r]
    latencies = sorted(a["latency_s"] for a in answers if a.get("latency_s"))
    report["calls"] = {
        "vlm_calls": len(answers),
        "errors": sum(a["error"] for a in answers),
        "json_unparsed": sum(not a["parsed"] and not a["error"] for a in answers),
        "latency_median_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 3) if latencies else None,
    }

    # A sample for a human to check: lenient wrong, or strict and lenient disagree.
    pool = [r for r in rows if "gt_nohint" in r and (not r["gt_nohint"]["lenient"] or r["gt_nohint"]["strict"] != r["gt_nohint"]["lenient"])]
    random.Random(0).shuffle(pool)
    sample = [
        {"id": r["id"], "category": r["category"], "definition": manifest[r["id"]]["definition"], "vlm": r["gt_nohint"]["name_en"], "vlm_zh": r["gt_nohint"]["name"], "lenient": r["gt_nohint"]["lenient"]}
        for r in pool[:50]
    ]
    out = path.with_name(path.stem + "-score.json")
    out.write_text(json.dumps({"report": report, "adjudication_sample": sample, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
