import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from src.nutrition_lookup.food_category_map import classify_food_name
    from src.segmentation_pipeline.model import FoodSegmentationModel
except ImportError:  # pragma: no cover
    from nutrition_lookup.food_category_map import classify_food_name
    from model import FoodSegmentationModel


# Project root directory
ROOT = Path(__file__).resolve().parents[2]


class SegmentationNutritionPipeline:
    def __init__(
        self,
        model_path,
        class_names,
        nutrition_lookup_path=None,
        category_profile_path=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FoodSegmentationModel(
            num_classes=len(class_names)
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.eval()
        self.class_names = class_names

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Default lookup files
        if nutrition_lookup_path is None:
            nutrition_lookup_path = (
                ROOT
                / "data"
                / "nutrition_db"
                / "food_lookup.json"
            )

        if category_profile_path is None:
            category_profile_path = (
                ROOT
                / "data"
                / "nutrition_db"
                / "category_profile.json"
            )

        # Helpful sanity checks
        if not nutrition_lookup_path.exists():
            raise FileNotFoundError(
                f"food_lookup.json not found at {nutrition_lookup_path}"
            )

        if not category_profile_path.exists():
            raise FileNotFoundError(
                f"category_profile.json not found at {category_profile_path}"
            )

        self.nutrition_lookup = json.loads(
            nutrition_lookup_path.read_text(encoding="utf-8")
        )

        self.category_profile = json.loads(
            category_profile_path.read_text(encoding="utf-8")
        )

    def _predict_mask(self, image):
        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().squeeze(0)
        return probs

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        probs = self._predict_mask(image)

        results = []

        for idx, class_name in enumerate(self.class_names):
            region_prob = float(probs[idx].max())

            if region_prob > 0.2:
                category = classify_food_name(class_name)

                results.append({
                    "class": class_name,
                    "category": category,
                    "confidence": region_prob,
                })

        if not results:
            return {
                "items": [],
                "estimated_nutrition": {},
            }

        nutrition_estimate = {}

        for item in results:
            category = item["category"]

            if category in self.category_profile:
                profile = self.category_profile[category]

                for nutrient, value in profile.items():
                    nutrition_estimate[nutrient] = (
                        nutrition_estimate.get(nutrient, 0.0)
                        + float(value) * item["confidence"]
                    )

        return {
            "items": results,
            "estimated_nutrition": nutrition_estimate,
        }