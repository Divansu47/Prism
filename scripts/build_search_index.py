import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "data" / "nutrition_db" / "food_lookup.json"

OUT = ROOT / "data" / "nutrition_db" / "search_index.json"


def normalize(text):

    return (
        text.lower()
            .replace("_", " ")
            .replace("-", " ")
            .strip()
    )


def tokenize(text):

    return normalize(text).split()


def main():

    with open(DB, "r", encoding="utf-8") as f:
        foods = json.load(f)

    index = defaultdict(set)

    for food in foods.keys():

        name = normalize(food)

        words = tokenize(name)

        index[name].add(name)

        if len(words) >= 2:

            for word in words:
                if len(word) > 2:
                    index[word].add(name)

            prefix = words[0]
            suffix = words[-1]

            index[prefix].add(name)
            index[suffix].add(name)

    final = {}

    for key, value in index.items():

        final[key] = sorted(value)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4)

    print(f"Foods        : {len(foods)}")
    print(f"Search Keys  : {len(final)}")
    print(f"Saved        : {OUT}")


if __name__ == "__main__":
    main()