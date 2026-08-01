import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "data" / "nutrition_db" / "food_lookup.json"
OUT = ROOT / "data" / "nutrition_db" / "canonical_map.json"


STOP_WORDS = {
    "raw",
    "fresh",
    "boiled",
    "fried",
    "roasted",
    "baked",
    "grilled",
    "steamed",
    "toasted",
    "dried",
    "frozen",
    "sweet",
    "salted",
    "whole",
    "low",
    "fat",
    "reduced",
    "skinless",
    "boneless",
    "with",
    "without",
    "in",
    "on",
    "and",
    "of",
    "the",
}


def normalize(text: str):

    return (
        text.lower()
            .replace("_", " ")
            .replace("-", " ")
            .strip()
    )


def canonical_name(food: str):

    words = normalize(food).split()

    meaningful = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    if not meaningful:
        return normalize(food)

    # Head noun is usually the last meaningful word.
    return meaningful[-1]


def main():

    with open(DB, "r", encoding="utf-8") as f:
        foods = json.load(f)

    mapping = {}

    for food in foods:

        mapping[normalize(food)] = canonical_name(food)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    print(f"Generated {len(mapping)} canonical mappings.")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()