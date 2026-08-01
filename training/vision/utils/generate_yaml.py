from pathlib import Path
import yaml


category_file = Path(
    "datasets/foodseg103/FoodSeg103/category_id.txt"
)

names = {}

with open(category_file, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        idx, name = line.split(maxsplit=1)

        idx = int(idx)

        if idx == 0:
            continue

        names[idx - 1] = name

config = {

    "path": "datasets/foodseg103_yolo",

    "train": "images/train",

    "val": "images/val",

    "task": "segment",

    "nc": len(names),

    "names": names
}

with open(
    "training/vision/foodseg.yaml",
    "w",
    encoding="utf-8"
) as f:

    yaml.dump(
        config,
        f,
        sort_keys=False,
        allow_unicode=True
    )

print("foodseg.yaml generated.")