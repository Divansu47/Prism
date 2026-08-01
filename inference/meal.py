from collections import defaultdict


# ---------------------------------------------------------
# Meal definitions
# ---------------------------------------------------------

MEALS = [
    {
        "name": "Burger",
        "required": {"bread", "steak"},
        "optional": {
            "lettuce",
            "tomato",
            "onion",
            "cheese",
            "sauce",
        },
    },
    {
        "name": "Pizza",
        "required": {"pizza"},
        "optional": {
            "cheese",
            "tomato",
            "pepperoni",
            "mushroom",
        },
    },
    {
        "name": "Fried Rice",
        "required": {"rice"},
        "optional": {
            "egg",
            "chicken",
            "vegetables",
        },
    },
    {
        "name": "Salad",
        "required": {"lettuce"},
        "optional": {
            "tomato",
            "cucumber",
            "onion",
        },
    },
]


# ---------------------------------------------------------
# Merge duplicate ingredients
# ---------------------------------------------------------

def merge_foods(foods):

    merged = {}

    for food in foods:

        name = food["class"].lower()

        if name not in merged:

            merged[name] = dict(food)

            continue

        merged[name]["pixel_area"] += food["pixel_area"]

        merged[name]["relative_volume"] += food["relative_volume"]

        merged[name]["avg_depth"] = (
            merged[name]["avg_depth"] + food["avg_depth"]
        ) / 2.0

        merged[name]["confidence"] = max(
            merged[name]["confidence"],
            food["confidence"],
        )

    return list(merged.values())


# ---------------------------------------------------------
# Guess meal
# ---------------------------------------------------------

def infer_meal(foods):

    ingredients = {food["class"].lower() for food in foods}

    best_name = "Mixed Meal"
    best_score = -1

    for meal in MEALS:

        if not meal["required"].issubset(ingredients):
            continue

        score = len(meal["required"])

        score += len(
            ingredients & meal["optional"]
        )

        if score > best_score:

            best_score = score
            best_name = meal["name"]

    return best_name


# ---------------------------------------------------------
# Sum nutrition
# ---------------------------------------------------------

def total_nutrition(food_cards):

    totals = defaultdict(float)

    for card in food_cards:

        nutrition = card.get("nutrition")

        if nutrition is None:
            continue

        for key, value in nutrition.items():

            try:
                totals[key] += float(value)
            except Exception:
                pass

    return dict(totals)


# ---------------------------------------------------------
# Total geometry
# ---------------------------------------------------------

def total_geometry(foods):

    volume = 0.0

    pixels = 0

    for food in foods:

        volume += food["relative_volume"]

        pixels += food["pixel_area"]

    return {
        "relative_volume": volume,
        "pixel_area": pixels,
    }