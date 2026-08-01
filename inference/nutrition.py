import json
from pathlib import Path

from .aliases import ALIASES
from .matcher import best_match, normalize

from .usda.cache import cached_search_best_match

ROOT = Path(__file__).resolve().parents[1]

DB_FILE = ROOT / "data" / "nutrition_db" / "food_lookup.json"
SEARCH_FILE = ROOT / "data" / "nutrition_db" / "search_index.json"
CANONICAL_FILE = ROOT / "data" / "nutrition_db" / "canonical_map.json"

MIN_ACCEPT_SCORE = 60


class NutritionEstimator:

    def __init__(self):

        with open(DB_FILE, "r", encoding="utf-8") as f:
            self.database = json.load(f)

        with open(SEARCH_FILE, "r", encoding="utf-8") as f:
            self.search_index = json.load(f)

        with open(CANONICAL_FILE, "r", encoding="utf-8") as f:
            self.canonical_map = json.load(f)

    def get_nutrition(self, detected_food):

        query = normalize(detected_food)

        # -------------------------
        # Alias resolution
        # -------------------------

        if query in ALIASES:
            query = ALIASES[query]
            method = "Alias"
            score = 100
        else:
            method = None
            score = 0

        # -------------------------
        # Direct lookup
        # -------------------------

        if query in self.database:

            matched = query

            if method is None:
                method = "Exact"
                score = 100

        else:

            matched, score, method = best_match(
                query,
                self.database,
                self.search_index
            )

            if matched is not None and score < MIN_ACCEPT_SCORE:
                matched = None

        if matched is None:

            return self._usda_fallback(detected_food, query)

        # -------------------------
        # Canonical mapping
        # -------------------------

        canonical = self.canonical_map.get(matched, matched)

        if canonical in self.database:
            matched = canonical

        item = self.database[matched]

        return {

            "query": detected_food,

            "matched_food": matched,

            "canonical_food": canonical,

            "match_method": method,

            "match_score": round(score, 2),

            "source": item["source"],

            "serving_basis": item["serving_basis"],

            "nutrition": item["nutrition"]

        }

    def _usda_fallback(self, detected_food, query):

        record = cached_search_best_match(query)

        if record is None:
            return None

        return {

            "query": detected_food,

            "matched_food": record["display_name"],

            "canonical_food": record["display_name"],

            "match_method": "USDA",

            "match_score": round(
                100.0 if record["is_generic"] else 80.0,
                2,
            ),

            "source": "usda",

            "serving_basis": "100g",

            "nutrition": {

                "calories": record["calories"],

                "protein": record["protein"],

                "fat": record["fat"],

                "carbs": record["carbs"],

                "fiber": record["fiber"],

            }

        }