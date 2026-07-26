import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from src.nutrition_lookup.food_category_map import classify_food_name
    from src.segmentation_pipeline.model import FoodSegmentationModel
except ImportError:  # pragma: no cover - allows direct module execution
    from nutrition_lookup.food_category_map import classify_food_name
    from model import FoodSegmentationModel


class SegmentationNutritionPipeline:
    def __init__(self, model_path, class_names, nutrition_lookup_path=None, category_profile_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FoodSegmentationModel(num_classes=len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        if nutrition_lookup_path is None:
            nutrition_lookup_path = Path(__file__).resolve().parents[1] / "data" / "nutrition_db" / "food_lookup.json"
        if category_profile_path is None:
            category_profile_path = Path(__file__).resolve().parents[1] / "data" / "nutrition_db" / "category_profile.json"

        self.nutrition_lookup = json.loads(Path(nutrition_lookup_path).read_text(encoding="utf-8"))
        self.category_profile = json.loads(Path(category_profile_path).read_text(encoding="utf-8"))

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
            return {"items": [], "estimated_nutrition": {}}

        nutrition_estimate = {}
        for item in results:
            category = item["category"]
            if category in self.category_profile:
                profile = self.category_profile[category]
                for nutrient, value in profile.items():
                    nutrition_estimate[nutrient] = nutrition_estimate.get(nutrient, 0.0) + float(value) * item["confidence"]

        return {"items": results, "estimated_nutrition": nutrition_estimate}
