import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw_food_data"

OUTPUT_DIR = ROOT / "data" / "nutrition_db"

FOOD_LOOKUP = OUTPUT_DIR / "food_lookup.json"

SCHEMA_FILE = OUTPUT_DIR / "nutrition_schema.json"


# --------------------------------------------
# Column rename map
# --------------------------------------------

COLUMN_MAP = {

    "Caloric Value": "Calories",

    "Fat( in g)": "Fat",

    "Saturated Fats( in g)": "Saturated Fat",

    "Monounsaturated Fats( in g)": "Monounsaturated Fat",

    "Polyunsaturated Fats( in g)": "Polyunsaturated Fat",

    "Carbohydrates( in g)": "Carbohydrates",

    "Sugars( in g)": "Sugar",

    "Protein( in g)": "Protein",

    "Dietary Fiber( in g)": "Fiber",

    "Cholesterol( in mg)": "Cholesterol",

    "Sodium( in mg)": "Sodium",

    "Water( in g)": "Water",

    "Vitamin A( in IU)": "Vitamin A",

    "Vitamin B1 (Thiamine)( in mg)": "Vitamin B1",

    "Vitamin B11 (Folic Acid/Folate)( in mcg)": "Folate",

    "Vitamin B12 (Cobalamine)( in mcg)": "Vitamin B12",

    "Vitamin B2 (Riboflavin)( in mg)": "Vitamin B2",

    "Vitamin B3 (Niacin)( in mg)": "Vitamin B3",

    "Vitamin B5 (Pantothenic Acid)( in mg)": "Vitamin B5",

    "Vitamin B6 ( in mg)": "Vitamin B6",

    "Vitamin C( in mg)": "Vitamin C",

    "Vitamin D( in IU)": "Vitamin D",

    "Vitamin E( in mg)": "Vitamin E",

    "Vitamin K( in mcg)": "Vitamin K",

    "Calcium( in mg)": "Calcium",

    "Copper( in mg)": "Copper",

    "Iron( in mg)": "Iron",

    "Magnesium( in mg)": "Magnesium",

    "Manganese( in mg)": "Manganese",

    "Phosphorus( in mg)": "Phosphorus",

    "Potassium( in mg)": "Potassium",

    "Selenium( in mcg)": "Selenium",

    "Zinc( in mg)": "Zinc",

    "Nutrition Density": "Nutrition Density"

}


UNITS = {

    "Calories": "kcal",

    "Fat": "g",

    "Saturated Fat": "g",

    "Monounsaturated Fat": "g",

    "Polyunsaturated Fat": "g",

    "Carbohydrates": "g",

    "Sugar": "g",

    "Protein": "g",

    "Fiber": "g",

    "Water": "g",

    "Cholesterol": "mg",

    "Sodium": "mg",

    "Vitamin A": "IU",

    "Vitamin B1": "mg",

    "Vitamin B2": "mg",

    "Vitamin B3": "mg",

    "Vitamin B5": "mg",

    "Vitamin B6": "mg",

    "Vitamin B12": "mcg",

    "Folate": "mcg",

    "Vitamin C": "mg",

    "Vitamin D": "IU",

    "Vitamin E": "mg",

    "Vitamin K": "mcg",

    "Calcium": "mg",

    "Copper": "mg",

    "Iron": "mg",

    "Magnesium": "mg",

    "Manganese": "mg",

    "Phosphorus": "mg",

    "Potassium": "mg",

    "Selenium": "mcg",

    "Zinc": "mg",

    "Nutrition Density": ""

}


food_database = {}

duplicates = 0


files = sorted(
    [
        f
        for f in RAW_DATA.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".csv"
        and "GROUP" in f.name.upper()
    ]
)
print("\nSearching CSV files...")

for f in files:
    print(f"✔ {f.name}")

print(f"\nFound {len(files)} csv files.\n")
print("=" * 60)
print("Building Nutrition Database")
print("=" * 60)

for file in files:

    print(f"\nReading {file.name}")

    df = pd.read_csv(file)

    df.columns = [c.strip() for c in df.columns]

    print(f"Rows : {len(df)}")

    for _, row in df.iterrows():

        if "food" not in row:

            continue

        food = str(row["food"]).strip().lower()

        if food == "" or food == "nan":

            continue

        if food in food_database:

            duplicates += 1
            continue

        nutrition = {}

        for col in df.columns:

            if col == "food":
                continue

            if col.startswith("Unnamed"):
                continue

            name = COLUMN_MAP.get(col, col)

            value = row[col]

            if pd.isna(value):

                value = None

            nutrition[name] = value

        food_database[food] = {

            "matched_name": food,

            "source": "Primary",

            "group": file.stem,

            "serving_basis": "100 g",

            "nutrition": nutrition

        }

print("\nSaving database...")

with open(FOOD_LOOKUP, "w", encoding="utf-8") as f:

    json.dump(

        food_database,

        f,

        indent=4,

        ensure_ascii=False

    )

with open(SCHEMA_FILE, "w", encoding="utf-8") as f:

    json.dump(

        UNITS,

        f,

        indent=4

    )

print("\n" + "=" * 60)

print(f"Foods             : {len(food_database)}")

print(f"Duplicates        : {duplicates}")

print(f"Nutrients         : {len(UNITS)}")

print(f"Saved             : {FOOD_LOOKUP}")

print("=" * 60)