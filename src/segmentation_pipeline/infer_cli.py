import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation_pipeline.infer import SegmentationNutritionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to the image to run inference on")
    parser.add_argument("--model_path", default="weights/segmentation_model.pth")
    parser.add_argument("--nutrition_lookup", default="data/nutrition_db/food_lookup.json")
    parser.add_argument("--category_profile", default="data/nutrition_db/category_profile.json")
    args = parser.parse_args()

    pipeline = SegmentationNutritionPipeline(
        model_path=args.model_path,
        class_names=["rice_staple", "curry_gravy", "vegetable", "protein"],
        nutrition_lookup_path=args.nutrition_lookup,
        category_profile_path=args.category_profile,
    )

    result = pipeline.predict(args.image_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
