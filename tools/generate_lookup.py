from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]

nutrition_dir = (
    ROOT
    / "data"
    / "nutrition_db"
    / "food_nutrition"
)

all_foods = set()

for csv in sorted(nutrition_dir.glob("FOOD-DATA-GROUP*.csv")):

    df = pd.read_csv(csv)

    all_foods.update(
        df["food"]
        .dropna()
        .str.lower()
        .str.strip()
        .tolist()
    )

print(f"{len(all_foods)} foods loaded.")

lookup = {}

detector_classes = [
    "apple",
    "banana",
    "orange",
    "tomato",
    "potato",
    "egg",
    "bread",
    "burger",
    "pizza",
    "sandwich",
    "hot dog",
    "rice",
    "broccoli",
    "carrot",
    "cucumber",
    "lettuce",
    "steak",
    "wine",
    "sauce",
]

for cls in detector_classes:

    matches = [
        x for x in all_foods
        if cls in x
    ]

    lookup[cls] = matches

with open(
    ROOT / "lookup_candidates.json",
    "w",
) as f:

    json.dump(
        lookup,
        f,
        indent=4,
    )

print("Saved lookup_candidates.json")