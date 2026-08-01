from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "lookup_candidates.json", "r") as f:
    candidates = json.load(f)

lookup = {}

preferred = {
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "tomato": "tomato",
    "potato": "potato",
    "egg": "scrambled eggs",
    "bread": "white bread",
    "burger": "hamburger",
    "pizza": "cheese pizza",
    "sandwich": "chicken sandwich",
    "hot dog": "hot dog",
    "rice": "white rice",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "cucumber": "cucumber",
    "lettuce": "lettuce",
    "steak": "beef steak",
    "wine": "red wine",
    "sauce": "tomato sauce",
}

for detector_class, target in preferred.items():

    if detector_class not in candidates:
        continue

    found = None

    for food in candidates[detector_class]:

        if food == target:
            found = food
            break

    if found is None and len(candidates[detector_class]):
        found = candidates[detector_class][0]

    lookup[detector_class] = found

output = (
    ROOT
    / "data"
    / "nutrition_db"
    / "food_lookup.json"
)

with open(output, "w") as f:
    json.dump(lookup, f, indent=4)

print(f"Saved {output}")