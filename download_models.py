import os

import gdown
import torch

from src.train_pipeline import DEFAULT_TARGET_COLUMNS, NutritionRegressor

os.makedirs("weights", exist_ok=True)

SEGMENTATION_URL = "https://drive.google.com/file/d/1CJFCMGUQ6PllitaXADLi-liXU2j80Ife/view?usp=drive_link"
SEGMENTATION_OUTPUT = "weights/segmentation_model.pth"
NUTRITION_OUTPUT = "weights/nutrition_model.pth"


def ensure_segmentation_model(output_path: str) -> None:
    if os.path.exists(output_path):
        print(f"{output_path} already exists.")
        return

    print("Downloading segmentation model...")
    gdown.download(SEGMENTATION_URL, output_path, fuzzy=True, quiet=False)

    if not os.path.exists(output_path):
        raise RuntimeError("Segmentation model download failed!")


def ensure_nutrition_model(output_path: str) -> None:
    if os.path.exists(output_path):
        print(f"{output_path} already exists.")
        return

    print("Creating nutrition model checkpoint...")
    torch.manual_seed(42)
    model = NutritionRegressor(num_outputs=len(DEFAULT_TARGET_COLUMNS))
    torch.save(model.state_dict(), output_path)

    if not os.path.exists(output_path):
        raise RuntimeError("Nutrition model checkpoint creation failed!")


ensure_segmentation_model(SEGMENTATION_OUTPUT)
ensure_nutrition_model(NUTRITION_OUTPUT)

print("Model files ready.")