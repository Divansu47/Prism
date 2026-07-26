from src.nutrition_lookup.food_category_map import classify_food_name


def test_classify_food_name_alias_matches_category_mapping():
    assert classify_food_name("White rice, cooked") == "rice_staple"
    assert classify_food_name("Chicken curry") == "curry_gravy"
    assert classify_food_name("Unknown snack") == "unmatched"
