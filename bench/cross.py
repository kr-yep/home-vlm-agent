"""Compare detector x VLM combinations, from the output of bench.score on each run.

The headline is end to end, over all objects: the detector boxed the object (IoU >= 0.5)
and the name shown for that box is right. Objects with no box count as misses, so a
detector cannot look good by boxing only the easy ones. The same run is also read at a
higher detector threshold, since fewer boxes means fewer VLM calls. That reading is
approximate: the eval kept each object's best-IoU box at 0.2, and a slightly worse box
above the threshold is not recovered.
    python -m bench.cross /data/runs yolo26m+12B=recognition yoloe+12B=recognition-12b-yoloe ...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SPLITS = ("coco", "non_coco", "all")
THRESHOLDS = (0.2, 0.3, 0.5)


def load(runs: Path, stem: str) -> dict[int, dict]:
    rows = {r["id"]: r for r in json.loads((runs / f"{stem}-score.json").read_text(encoding="utf-8"))["rows"]}
    for line in (runs / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows[record["id"]]["yolo_conf"] = record.get("yolo_conf")
    return rows


def hit(row: dict, key: str, measure: str = "lenient", threshold: float = 0.0) -> bool:
    if key != "gt_nohint" and (not row["yolo_matched"] or (row.get("yolo_conf") or 0) < threshold):
        return False
    return bool((row.get(key) or {}).get(measure))


def percent(values) -> str:
    values = list(values)
    return f"{sum(values) / len(values):.1%}"


def sign_test(a: dict, b: dict, ids, test) -> str:
    only_a = sum(test(a[i]) and not test(b[i]) for i in ids)
    only_b = sum(test(b[i]) and not test(a[i]) for i in ids)
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n) if n else 1.0
    return f"{only_a} vs {only_b}, p={p:.2g}"


def main() -> None:
    runs = Path(sys.argv[1])
    cells = {name: load(runs, stem) for name, stem in (arg.split("=") for arg in sys.argv[2:])}
    report = {}
    for name, rows in cells.items():
        section = {}
        for split in SPLITS:
            sel = [r for r in rows.values() if split == "all" or r["split"] == split]
            found = [r for r in sel if r["yolo_matched"]]
            section[split] = {
                "objects": len(sel),
                "boxed": percent(r["yolo_matched"] for r in sel),
                "end to end, VLM with hint": percent(hit(r, "yolo_hint") for r in sel),
                "end to end, VLM no hint": percent(hit(r, "yolo_nohint") for r in sel),
                "end to end, detector's own name": percent(r["yolo_lenient"] for r in sel),
                "on boxed objects, VLM with hint": percent(hit(r, "yolo_hint") for r in found),
                "on boxed objects, VLM no hint": percent(hit(r, "yolo_nohint") for r in found),
                "on boxed objects, COCO-level, VLM with hint": percent(hit(r, "yolo_hint", "coarse") for r in found),
                "VLM on the true box": percent(hit(r, "gt_nohint") for r in sel),
            }
        section["end to end with hint, by detector threshold"] = {
            str(t): percent(hit(r, "yolo_hint", threshold=t) for r in rows.values()) for t in THRESHOLDS
        }
        latencies = sorted(r[k]["latency_s"] for r in rows.values() for k in ("gt_nohint", "yolo_hint", "yolo_nohint") if (r.get(k) or {}).get("latency_s"))
        section["vlm_latency_median_s"] = latencies[len(latencies) // 2] if latencies else None
        section["vlm_errors"] = sum((r.get(k) or {}).get("error", False) for r in rows.values() for k in ("gt_nohint", "yolo_hint", "yolo_nohint"))
        report[name] = section
    names = list(cells)
    ids = sorted(set.intersection(*(set(c) for c in cells.values())))
    report["paired sign tests, all objects (first vs second)"] = {
        f"{a} vs {b}": {
            "end to end, hint": sign_test(cells[a], cells[b], ids, lambda r: hit(r, "yolo_hint")),
            "end to end, no hint": sign_test(cells[a], cells[b], ids, lambda r: hit(r, "yolo_nohint")),
            "true box": sign_test(cells[a], cells[b], ids, lambda r: hit(r, "gt_nohint")),
        }
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    (runs / "cross.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
