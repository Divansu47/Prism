from pathlib import Path
import re
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

USDA_DIR = ROOT / "data" / "usda" / "FoodData_Central_csv_2026-04-30"

OUTPUT_DIR = ROOT / "data" / "nutrition_db"

OUTPUT_DB = OUTPUT_DIR / "food_lookup_usda.db"

FOOD_NUTRIENT_CHUNK_SIZE = 500_000

IMPORTANT_NUTRIENTS = {
    1008: "calories",
    1003: "protein",
    1004: "fat",
    1005: "carbs",
    1079: "fiber",
}

DATA_TYPE_PRIORITY = {
    "foundation_food": 0,
    "sr_legacy_food": 1,
    "survey_fndds_food": 2,
    "experimental_food": 3,
    "branded_food": 4,
}


def normalize_name(name):

    name = str(name).strip().lower()

    name = re.sub(r"[^a-z0-9\s]", " ", name)

    name = re.sub(r"\s+", " ", name).strip()

    return name


def score_for_data_type(data_type):

    dtype = str(data_type).strip().lower()

    return DATA_TYPE_PRIORITY.get(dtype, 99)


def create_schema(conn):

    print("Creating schema...")

    conn.execute("DROP TABLE IF EXISTS foods")

    conn.execute(
        """
        CREATE TABLE foods (
            fdc_id INTEGER PRIMARY KEY,
            display_name TEXT,
            normalized_name TEXT,
            data_type TEXT,
            brand_owner TEXT,
            is_generic INTEGER,
            calories REAL,
            protein REAL,
            fat REAL,
            carbs REAL,
            fiber REAL,
            publication_date TEXT,
            score INTEGER
        )
        """
    )

    conn.commit()


def create_indexes(conn):

    print("Creating indexes...")

    conn.execute(
        "CREATE INDEX idx_normalized_name ON foods(normalized_name)"
    )

    conn.execute(
        "CREATE INDEX idx_data_type ON foods(data_type)"
    )

    conn.commit()


def load_food_table():

    path = USDA_DIR / "food.csv"

    print("Loading food.csv...")

    food = pd.read_csv(
        path,
        usecols=[
            "fdc_id",
            "description",
            "data_type",
            "publication_date",
        ],
        dtype={"fdc_id": "int32"},
        low_memory=False,
    )

    print(f"Loaded {len(food):,} food rows.\n")

    return food


def load_branded_owner_map():

    path = USDA_DIR / "branded_food.csv"

    if not path.exists():
        return {}

    print("Loading branded_food.csv...")

    branded = pd.read_csv(
        path,
        usecols=["fdc_id", "brand_owner"],
        dtype={"fdc_id": "int32"},
        low_memory=False,
    )

    owner_map = dict(zip(branded["fdc_id"], branded["brand_owner"]))

    print(f"Loaded {len(owner_map):,} brand owner entries.\n")

    return owner_map


def build_nutrient_map():

    path = USDA_DIR / "food_nutrient.csv"

    print("Loading food_nutrient.csv in chunks...")

    ids = set(IMPORTANT_NUTRIENTS.keys())

    nutrient_map = {}

    reader = pd.read_csv(
        path,
        usecols=["fdc_id", "nutrient_id", "amount"],
        dtype={
            "fdc_id": "int32",
            "nutrient_id": "int32",
            "amount": "float32",
        },
        chunksize=FOOD_NUTRIENT_CHUNK_SIZE,
        low_memory=False,
    )

    matched_rows = 0

    for chunk_idx, chunk in enumerate(reader, start=1):

        chunk = chunk[chunk["nutrient_id"].isin(ids)]

        for fdc_id, nutrient_id, amount in zip(
            chunk["fdc_id"],
            chunk["nutrient_id"],
            chunk["amount"],
        ):

            if pd.isna(amount):
                continue

            key = IMPORTANT_NUTRIENTS[int(nutrient_id)]

            entry = nutrient_map.setdefault(int(fdc_id), {})

            entry[key] = float(amount)

        matched_rows += len(chunk)

        print(f"Chunk {chunk_idx} processed | matched rows so far: {matched_rows:,}")

    print(f"\nNutrient map built for {len(nutrient_map):,} foods.\n")

    return nutrient_map


def build_records(food, nutrient_map, brand_owner_map):

    print("Building records...")

    records = []

    for row in food.itertuples(index=False):

        fdc_id = int(row.fdc_id)

        description = row.description

        data_type = row.data_type

        publication_date = row.publication_date

        dtype_lower = str(data_type).strip().lower()

        nutrients = nutrient_map.get(fdc_id, {})

        score = score_for_data_type(data_type)

        is_generic = 0 if dtype_lower == "branded_food" else 1

        brand_owner = brand_owner_map.get(fdc_id)

        if pd.isna(brand_owner):
            brand_owner = None

        records.append(
            (
                fdc_id,
                description,
                normalize_name(description),
                data_type,
                brand_owner,
                is_generic,
                nutrients.get("calories", 0.0),
                nutrients.get("protein", 0.0),
                nutrients.get("fat", 0.0),
                nutrients.get("carbs", 0.0),
                nutrients.get("fiber", 0.0),
                publication_date,
                score,
            )
        )

    print(f"Built {len(records):,} raw records.\n")

    return records


def deduplicate_records(records):

    print("Deduplicating records...")

    # Key on (normalized_name, is_generic) so generic entries
    # (Foundation/SR Legacy/Survey/Experimental) dedupe against
    # each other by priority score, while branded foods stay
    # separately searchable and are never merged into a generic row.
    best = {}

    for rec in records:

        normalized_name = rec[2]
        is_generic = rec[5]
        publication_date = rec[11]
        score = rec[12]

        key = (normalized_name, is_generic)

        if key not in best:
            best[key] = rec
            continue

        current = best[key]
        current_pub_date = current[11]
        current_score = current[12]

        if score < current_score:
            best[key] = rec
        elif score == current_score:
            if str(publication_date) > str(current_pub_date):
                best[key] = rec

    deduped = list(best.values())

    print(f"Deduplicated to {len(deduped):,} records.\n")

    return deduped


def insert_records(conn, records):

    print("Inserting records into SQLite...")

    conn.executemany(
        """
        INSERT INTO foods (
            fdc_id,
            display_name,
            normalized_name,
            data_type,
            brand_owner,
            is_generic,
            calories,
            protein,
            fat,
            carbs,
            fiber,
            publication_date,
            score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    conn.commit()

    print(f"Inserted {len(records):,} rows.\n")


def main():

    print("=" * 60)
    print("USDA SQLite Database Builder")
    print("=" * 60)
    print()

    print("ROOT =", ROOT)
    print("USDA_DIR =", USDA_DIR)
    print("Exists =", USDA_DIR.exists())
    print("Food exists =", (USDA_DIR / "food.csv").exists())
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    food = load_food_table()

    brand_owner_map = load_branded_owner_map()

    nutrient_map = build_nutrient_map()

    records = build_records(food, nutrient_map, brand_owner_map)

    records = deduplicate_records(records)

    conn = sqlite3.connect(OUTPUT_DB)

    try:

        create_schema(conn)

        insert_records(conn, records)

        create_indexes(conn)

    finally:

        conn.close()

    print("=" * 60)
    print("USDA database created successfully")
    print(f"Foods : {len(records):,}")
    print(f"Output: {OUTPUT_DB}")
    print("=" * 60)
    print()

    print("Finished successfully.")


if __name__ == "__main__":

    main()