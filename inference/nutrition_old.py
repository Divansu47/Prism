from pathlib import Path
import json

from rapidfuzz import process, fuzz


class NutritionEstimator:

    # Normalize YOLO class names to canonical search terms
    NORMALIZE = {
        "burger": "hamburger",
        "hamburger": "hamburger",
        "cheeseburger": "cheeseburger",

        "bread": "bread",
        "toast": "bread",

        "egg": "egg",

        "rice": "rice",

        "pizza": "pizza",

        "potato": "potato",

        "broccoli": "broccoli",

        "steak": "steak",

        "wine": "wine",

        "hot dog": "hotdog",
        "hotdog": "hotdog",

        "fries": "french fries",
        "french fries": "french fries",

        "chips": "potato chips",
    }
    PREFERRED = {
    "bread": {"white", "wheat", "whole", "plain"},
    "egg": {"raw", "boiled", "cooked", "poached"},
    "rice": {"white", "brown", "cooked", "boiled", "plain"},
    "potato": {"raw", "cooked", "baked"},
    "broccoli": {"raw", "cooked"},
    "pizza": {"cheese", "pepperoni"},
    "burger": {"hamburger", "cheeseburger"},
    "steak": {"sirloin", "beef", "rib", "t-bone", "porterhouse"},
    "wine": {"red", "white", "table"},
    }

    PENALTY = {
        "mix",
        "crumb",
        "crumbs",
        "bran",
        "flour",
        "sake",
        "roll",
        "sandwich",
        "burger king",
        "mcdonalds",
        "pizza hut",
        "dominos",
        "casserole",
        "salad",
        "soup",
        "sauce",
        "gravy",
        "bacon",
        "cheese sauce",
        "noodles",
    }

    def __init__(self):

        root = Path(__file__).resolve().parents[1]

        lookup_path = (
            root
            / "data"
            / "nutrition_db"
            / "food_lookup.json"
        )

        with open(lookup_path, "r", encoding="utf-8") as f:
            self.food_db = json.load(f)

        self.food_names = list(self.food_db.keys())

    def _normalize(self, food):

        food = food.lower().strip()

        return self.NORMALIZE.get(food, food)

    def find_food(self, yolo_food):

        query = self._normalize(yolo_food)

        # Exact match
        if query in self.food_db:
            return query

        candidates = []

        for food in self.food_names:

            name = food.lower()

            if (
                query in name
                or name.startswith(query)
                or name.endswith(query)
            ):
                candidates.append(food)

        if not candidates:
            candidates = self.food_names

        # Words indicating processed/restaurant foods
        penalty_words = {
            "burger king",
            "mcdonalds",
            "pizza hut",
            "dominos",
            "subway",
            "kfc",
            "wendys",
            "carls",
            "jack",
            "taco",
            "sandwich",
            "roll",
            "casserole",
            "salad",
            "soup",
            "bacon",
            "cheese sauce",
            "sauce",
            "gravy",
            "noodles",
            "with",
        }

        

        best_food = None
        best_score = float("-inf")

        preferred_words = self.PREFERRED.get(yolo_food.lower(), set())

        for food in candidates:

            name = food.lower()

            score = fuzz.token_set_ratio(query, name)

            # Prefer shorter names
            score -= len(name) * 0.35

            words = set(name.replace("-", " ").split())

            # Exact word bonus
            if query in words:
                score += 40

            # Starts with query
            if name.startswith(query):
                score += 20

            # Generic names
            if len(words) <= 2:
                score += 20

            # Preferred descriptors
            if preferred_words & words:
                score += 50

            # Penalize processed foods
            for bad in self.PENALTY:
                if bad in name:
                    score -= 40

            if score > best_score:
                best_score = score
                best_food = food

        return best_food

    def get_nutrition(self, yolo_food):

        matched = self.find_food(yolo_food)

        if matched is None:
            return None

        return {
            "matched_food": matched,
            "nutrition": self.food_db[matched]
        }