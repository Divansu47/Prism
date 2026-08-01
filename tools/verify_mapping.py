from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]

mapping = json.load(open(ROOT / "data" / "nutrition_db" / "class_mapping.json"))

csvs = sorted(
    (ROOT / "data" / "nutrition_db" / "food_nutrition").glob("FOOD-DATA-GROUP*.csv")
)

foods = set()

for csv in csvs:
    df = pd.read_csv(csv)
    foods.update(df["food"].dropna().str.lower().str.strip())

print("=" * 60)

for detector, mapped in mapping.items():

    mapped = mapped.lower()

    if mapped in foods:
        print(f"✓ {detector:15} -> {mapped}")
    else:
        print(f"✗ {detector:15} -> {mapped}")
        suggestions = [f for f in foods if detector in f][:10]

        if suggestions:
            print("   Suggestions:")
            for s in suggestions:
                print("    ", s)

print("=" * 60)