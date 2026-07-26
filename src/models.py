import torch.nn as nn
from torchvision.models import resnet18

DEFAULT_TARGET_COLUMNS = [
    "Caloric Value",
    "Fat( in g)",
    "Saturated Fats( in g)",
    "Monounsaturated Fats( in g)",
    "Polyunsaturated Fats( in g)",
    "Carbohydrates( in g)",
    "Sugars( in g)",
    "Protein( in g)",
    "Dietary Fiber( in g)",
    "Cholesterol( in mg)",
    "Sodium( in g)",
    "Water( in g)",
    "Vitamin A( in mg)",
    "Vitamin B1 (Thiamine)( in mg)",
    "Vitamin B11 (Folic Acid)( in mg)",
    "Vitamin B12( in mg)",
    "Vitamin B2 (Riboflavin)( in mg)",
    "Vitamin B3 (Niacin)( in mg)",
    "Vitamin B5 (Pantothenic Acid)( in mg)",
    "Vitamin B6( in mg)",
    "Vitamin C( in mg)",
    "Vitamin D( in mg)",
    "Vitamin E( in mg)",
    "Vitamin K( in mg)",
    "Calcium( in mg)",
    "Copper( in mg)",
    "Iron( in mg)",
    "Magnesium( in mg)",
    "Manganese( in mg)",
    "Phosphorus( in mg)",
    "Potassium( in mg)",
    "Selenium( in mg)",
    "Zinc( in mg)",
    "Nutrition Density",
]


class NutritionRegressor(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.backbone(x)