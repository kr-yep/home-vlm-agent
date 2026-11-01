"""Which name should the app show: YOLO's, the VLM's, or a rule over both?

Offline, from recognition.jsonl (VLM answers on YOLO's boxes, with the hint).
Every rule is judged with the same yardstick: right if the shown name is in the
object's COCO-class group or WordNet agrees (score.coarse / yolo_correct).
The confidence threshold is fixed in advance (0.5, the middle of YOLO's range),
not tuned on these objects; other values are printed only to show sensitivity.
    uv run --frozen --only-group bench python -m bench.fusion /data/runs/recognition.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .coco_lvis import COCO_TO_LVIS, yolo_correct
from .score import coarse, lenient, strict

THRESHOLD = 0.5


def vlm_agrees_with_yolo(name: str, yolo_class: str) -> bool:
    """The VLM's answer stays inside YOLO's class ("MacBook" for laptop counts)."""

    group = {yolo_class} | {g.split("_(")[0] for g in COCO_TO_LVIS.get(yolo_class, ())}
    return any(strict(name, {"synonyms": [g], "category": g}) for g in group)


def main() -> None:
    path = Path(sys.argv[1])
    manifest = {o["id"]: o for o in json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "/data/data/lvis-eval/manifest.json").read_text())}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def yolo_right(r):
        return yolo_correct(r["yolo_class"], r["category"]) or lenient(r["yolo_class"], manifest[r["id"]])

    def vlm_right(r):
        name = r["yolo_hint"].get("name_en", "")
        obj = manifest[r["id"]]
        return bool(name) and (lenient(name, obj) or coarse(name, obj))

    rules = {
        "YOLO only": lambda r, t: yolo_right(r),
        "VLM always (current app)": lambda r, t: vlm_right(r),
        "YOLO if conf>=t else VLM": lambda r, t: yolo_right(r) if r["yolo_conf"] >= t else vlm_right(r),
        "VLM if it agrees with YOLO or conf<t, else YOLO": lambda r, t: vlm_right(r)
        if (r["yolo_conf"] < t or vlm_agrees_with_yolo(r["yolo_hint"].get("name_en", ""), r["yolo_class"]))
        else yolo_right(r),
    }
    found = [r for r in records if r["yolo_class"] is not None and "yolo_hint" in r]
    report = {"threshold": THRESHOLD, "objects": len(records), "yolo_found": len(found)}
    for split in ("coco", "non_coco", "all"):
        sel = [r for r in found if split == "all" or r["split"] == split]
        report[split] = {"found": len(sel)}
        for name, rule in rules.items():
            hits = sum(rule(r, THRESHOLD) for r in sel)
            report[split][name] = f"{hits / len(sel):.1%} ({hits}/{len(sel)})" if sel else "n/a"
    report["sensitivity (all found)"] = {
        f"t={t}": {name: f"{sum(rule(r, t) for r in found) / len(found):.1%}" for name, rule in list(rules.items())[2:]}
        for t in (0.3, 0.4, 0.5, 0.6, 0.7)
    }
    report["yolo_conf_quartiles"] = sorted(r["yolo_conf"] for r in found)[:: max(1, len(found) // 4)]
    out = path.with_name(path.stem + "-fusion.json")
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
