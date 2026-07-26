"""
food_category_map.py

Defines the segmentation category taxonomy and a keyword-based mapping
from Kaggle "Food Nutrition Dataset" food names to those categories.

Why this exists:
The segmentation model (Phase 2) predicts pixel classes like "rice_staple"
or "vegetable". The Kaggle CSV has individual food names like "White rice,
cooked" or "Spinach, boiled". We need a many-to-one mapping: many CSV rows
-> one segmentation category, so that when the segmenter says "this region
is rice_staple", we know which CSV rows to average / sample from for
nutrition-per-100g.

Adjust CATEGORY_KEYWORDS to match your actual bowl contents. Keywords are
matched against the lowercased, whitespace-normalized food name.
"""

CATEGORIES = [
    "background",
    "rice_staple",
    "curry_gravy",
    "vegetable",
    "protein",
]

# Order matters: first matching category wins. Put more specific terms
# (e.g. "chicken curry") before generic ones (e.g. "chicken").
CATEGORY_KEYWORDS = {
    "rice_staple": [
        "rice", "biryani", "pulao", "khichdi", "roti", "chapati", "naan",
        "bread", "noodle", "pasta", "quinoa", "wheat", "oats", "oatmeal",
        "barley", "millet", "couscous",
    ],
    "curry_gravy": [
        "curry", "dal", "lentil", "gravy", "sambar", "rasam", "stew",
        "soup",
    ],
    "protein": [
        "chicken", "egg", "fish", "paneer", "tofu", "beef", "pork",
        "mutton", "shrimp", "prawn", "lamb", "meat", "steak", "turkey",
        "sausage", "bacon", "ham", "salmon", "tuna", "trout", "cod",
        "flounder", "halibut", "snapper", "sheepshead", "scallop",
        "crab", "lobster", "clam", "oyster", "mussel", "duck", "veal",
        "sardine", "anchovy", "mackerel", "tilapia", "catfish",
    ],
    "vegetable": [
        "spinach", "vegetable", "cabbage", "carrot", "broccoli",
        "cauliflower", "peas", "beans", "potato", "salad", "greens",
        "okra", "pumpkin", "gourd", "brinjal", "eggplant", "beet",
        "corn", "zucchini", "pepper", "onion", "tomato", "cucumber",
        "radish", "turnip", "squash", "asparagus", "celery", "kale",
        "lettuce", "mushroom",
    ],
}


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def map_food_to_category(food_name: str) -> str:
    """Return the segmentation category for a given raw Kaggle food name.

    IMPORTANT: returns "unmatched" if no keyword hits -- do NOT silently
    default to a real category (e.g. "vegetable"). Your source dataset
    likely contains many foods irrelevant to bowl meals (candy, soda,
    packaged snacks). Silently bucketing those into "vegetable" would
    contaminate that category's average nutrition profile. Unmatched
    foods are reported separately by clean_nutrition_db.py so you can
    decide whether to add keywords for real bowl items you're missing,
    or just leave the irrelevant ones excluded.
    """
    name = normalize_name(food_name)
    for category in ["rice_staple", "curry_gravy", "protein", "vegetable"]:
        for kw in CATEGORY_KEYWORDS[category]:
            if kw in name:
                return category
    return "unmatched"


def classify_food_name(food_name: str) -> str:
    """Compatibility wrapper used by the segmentation inference pipeline."""
    return map_food_to_category(food_name)