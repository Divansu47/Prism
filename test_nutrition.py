from inference.nutrition import NutritionEstimator

nutrition = NutritionEstimator()

foods = [
    "burger",
    "pizza",
    "bread",
    "egg",
    "rice",
    "potato",
    "broccoli",
    "steak",
    "wine",
    "hot dog",
]

for food in foods:

    result = nutrition.get_nutrition(food)

    print("=" * 70)

    print("YOLO Prediction :", food)

    if result is None:
        print("No Match Found")
        continue

    print("Matched Food    :", result["matched_food"])

    print("Nutrition")

    for key, value in result["nutrition"].items():
        print(f"  {key:25}: {value}")