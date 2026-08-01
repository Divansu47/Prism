import json
from pathlib import Path


class NutritionLookup:

    def __init__(self):

        root = Path(__file__).resolve().parents[2]

        nutrition_dir = root / "data" / "nutrition_db"

        with open(nutrition_dir / "food_lookup.json", "r", encoding="utf-8") as f:
            self.foods = json.load(f)

        with open(nutrition_dir / "category_profile.json", "r", encoding="utf-8") as f:
            self.categories = json.load(f)

        with open(nutrition_dir / "serving_reference.json", "r", encoding="utf-8") as f:
            self.servings = json.load(f)

    def find_food(self, name):

        return self.foods.get(name.lower().strip())

    def find_category(self, category):

        return self.categories.get(category.lower().strip())

    def get_reference_mass(self, name):

        return self.servings.get(name.lower().strip())