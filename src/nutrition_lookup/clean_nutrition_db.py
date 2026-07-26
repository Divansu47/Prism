"""
clean_nutrition_db.py

Phase 1: Clean the Kaggle "Food Nutrition Dataset" CSV and build two
lookup artifacts used at inference time:

1. food_lookup.json
   { normalized_food_name: {nutrient_col: value, ...} }
   -> exact per-100g nutrition for any food name that appears in the CSV.

2. category_profile.json
   { category_name: {nutrient_col: avg_value, ...} }
   -> per-100g nutrition AVERAGED across all foods in that category.
   Used as the nutrition source once the segmentation model has assigned
   a pixel region to a category (we don't know the *exact* dish, only
   the category, so we use the category's typical nutrition profile).

Usage:
    python clean_nutrition_db.py --csv data/nutrition_db/food_nutrition.csv \
                                  --out data/nutrition_db/

Run this once you've placed the actual Kaggle CSV at the given path.
Column names below match the dataset's documented schema exactly --
if your downloaded file uses slightly different headers, this script
will print a clear error telling you which columns are missing.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

from food_category_map import map_food_to_category, normalize_name

# Nutrient columns we care about for downstream bowl-level prediction.
# Extend this list if you want to surface more of the ~30 micronutrient
# columns later -- keeping it lean for now keeps the JSON small and the
# API response simple for your teammate's frontend.
NUTRIENT_COLUMNS = [
    "Caloric Value",
    "Fat( in g)",
    "Saturated Fats( in g)",
    "Carbohydrates( in g)",
    "Sugars( in g)",
    "Protein( in g)",
    "Dietary Fiber( in g)",
    "Sodium( in g)",
]

FOOD_NAME_COLUMN = "Food"
COLUMN_ALIASES = {
    "food": FOOD_NAME_COLUMN,
    "Fat": "Fat( in g)",
    "Saturated Fats": "Saturated Fats( in g)",
    "Carbohydrates": "Carbohydrates( in g)",
    "Sugars": "Sugars( in g)",
    "Protein": "Protein( in g)",
    "Dietary Fiber": "Dietary Fiber( in g)",
    "Sodium": "Sodium( in g)",
}


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        raw_name: normalized_name
        for raw_name, normalized_name in COLUMN_ALIASES.items()
        if raw_name in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_input_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if path.is_dir():
        csv_files = sorted(path.glob("FOOD-DATA-GROUP*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No FOOD-DATA-GROUP*.csv files found in {path}")
        frames = [pd.read_csv(csv_file) for csv_file in csv_files]
        print(f"Merging {len(csv_files)} file(s): {[csv_file.name for csv_file in csv_files]}")
        df = pd.concat(frames, ignore_index=True)
    else:
        patterns = [csv_path] if any(ch in csv_path for ch in "*?[") else [csv_path]
        paths = []
        for pattern in patterns:
            matched = glob.glob(pattern)
            if matched:
                paths.extend(matched)
        if not paths:
            raise FileNotFoundError(f"No files matched: {csv_path}")
        frames = [pd.read_csv(p) for p in paths]
        print(f"Merging {len(paths)} file(s): {[os.path.basename(p) for p in paths]}")
        df = pd.concat(frames, ignore_index=True)
    return _prepare_dataframe(df)


def clean(csv_path: str, out_dir: str) -> None:
    df = load_input_dataframe(csv_path)

    missing = [c for c in [FOOD_NAME_COLUMN] + NUTRIENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Actual columns found: {list(df.columns)}\n"
            "Update FOOD_NAME_COLUMN / NUTRIENT_COLUMNS in this script to match."
        )

    df = df.dropna(subset=[FOOD_NAME_COLUMN])
    for col in NUTRIENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["_norm_name"] = df[FOOD_NAME_COLUMN].apply(normalize_name)
    df["_category"] = df[FOOD_NAME_COLUMN].apply(map_food_to_category)

    # 1. Per-food exact lookup (last occurrence wins on duplicate names)
    food_lookup = {}
    for _, row in df.iterrows():
        food_lookup[row["_norm_name"]] = {
            col: round(float(row[col]), 3) for col in NUTRIENT_COLUMNS
        }

    # 2. Per-category average profile -- EXCLUDE "unmatched" from this,
    # those foods stay in food_lookup.json (exact lookup still works)
    # but must never pollute a real category's average.
    category_profile = {}
    for category, group in df.groupby("_category"):
        if category == "unmatched":
            continue
        category_profile[category] = {
            col: round(float(group[col].mean()), 3) for col in NUTRIENT_COLUMNS
        }
        category_profile[category]["_num_foods_averaged"] = int(len(group))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "food_lookup.json"), "w") as f:
        json.dump(food_lookup, f, indent=2)
    with open(os.path.join(out_dir, "category_profile.json"), "w") as f:
        json.dump(category_profile, f, indent=2)

    print(f"Loaded {len(df)} food rows.")
    print(f"Built food_lookup.json with {len(food_lookup)} entries.")
    print("\nCategory coverage (used for category_profile.json averages):")
    for cat, prof in category_profile.items():
        print(f"  {cat}: {prof['_num_foods_averaged']} foods averaged")

    unmatched = df[df["_category"] == "unmatched"]
    print(f"\nUnmatched (excluded from category averages): {len(unmatched)} / {len(df)} "
          f"({100*len(unmatched)/len(df):.1f}%)")
    if len(unmatched) > 0:
        sample = unmatched[FOOD_NAME_COLUMN].head(15).tolist()
        print(f"Sample unmatched food names (review these -- add keywords for any real bowl items):")
        for name in sample:
            print(f"    {name}")

    missing_categories = set(["rice_staple", "curry_gravy", "vegetable", "protein"]) - set(category_profile.keys())
    if missing_categories:
        print(f"WARNING: no CSV rows matched these categories: {missing_categories}. "
              f"Add more keywords in food_category_map.py or manually seed values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the raw Kaggle CSV")
    parser.add_argument("--out", required=True, help="Output directory for the JSON lookups")
    args = parser.parse_args()
    clean(args.csv, args.out)
