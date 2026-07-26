import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_pipeline import NutritionRegressor


def test_model_output_shape():
    model = NutritionRegressor(num_outputs=8)
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 8)
