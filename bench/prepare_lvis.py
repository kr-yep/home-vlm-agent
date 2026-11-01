"""Pick an indoor evaluation set from LVIS v1 val and fetch its COCO images.

Needs network (COCO image server), so it runs outside the internal network:
    docker run --rm -v /root/.cache/home-vlm/data:/data home-vlm-app python -m bench.prepare_lvis

Writes /data/lvis-eval/manifest.json and /data/lvis-eval/images/*.jpg.
LVIS labels are federated (not exhaustive), so the set is a list of annotated
objects to name, not a detection benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .coco_lvis import COCO_TO_LVIS, INDOOR_ANCHORS, coco_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvis", default="/data/lvis_v1_val.json")
    parser.add_argument("--out", default="/data/lvis-eval")
    parser.add_argument("--per-split", type=int, default=250)
    parser.add_argument("--min-side", type=int, default=32)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lvis = json.loads(Path(args.lvis).read_text())
    categories = {c["id"]: c for c in lvis["categories"]}
    names = {c["name"] for c in categories.values()}
    unknown = sorted(n for group in COCO_TO_LVIS.values() for n in group if n not in names) + sorted(INDOOR_ANCHORS - names)
    if unknown:
        raise SystemExit(f"not LVIS category names: {unknown}")

    by_image = defaultdict(list)
    for annotation in lvis["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    images = {i["id"]: i for i in lvis["images"]}
    indoor = [
        image_id
        for image_id, annotations in by_image.items()
        if any(categories[a["category_id"]]["name"] in INDOOR_ANCHORS for a in annotations)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(indoor)

    picked, per_category, per_split = [], Counter(), Counter()
    for image_id in indoor:
        annotations = by_image[image_id][:]
        rng.shuffle(annotations)
        taken = 0
        for annotation in annotations:
            category = categories[annotation["category_id"]]
            x, y, w, h = annotation["bbox"]
            split = coco_split(category["name"])
            if (
                min(w, h) < args.min_side
                or category["name"] == "person"
                or per_category[category["name"]] >= args.per_category
                or per_split[split] >= args.per_split
                or taken >= args.per_image
            ):
                continue
            picked.append(
                {
                    "id": annotation["id"],
                    "image_id": image_id,
                    "file": f"{image_id:012d}.jpg",
                    "bbox_xyxy": [round(x), round(y), round(x + w), round(y + h)],
                    "category": category["name"],
                    "synonyms": category["synonyms"],
                    "synset": category["synset"],
                    "definition": category["def"],
                    "frequency": category["frequency"],
                    "split": split,
                }
            )
            per_category[category["name"]] += 1
            per_split[split] += 1
            taken += 1
        if all(per_split[s] >= args.per_split for s in ("coco", "non_coco")):
            break

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for image_id in sorted({p["image_id"] for p in picked}):
        target = out / "images" / f"{image_id:012d}.jpg"
        if not target.exists():
            urllib.request.urlretrieve(images[image_id]["coco_url"], target)
    (out / "manifest.json").write_text(json.dumps(picked, indent=1, ensure_ascii=False))
    print(
        f"{len(picked)} objects, {len({p['image_id'] for p in picked})} images, "
        f"{len(per_category)} categories, splits {dict(per_split)}"
    )


if __name__ == "__main__":
    main()
