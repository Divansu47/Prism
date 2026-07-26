import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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


class FoodImageDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        target = self.targets[idx]
        image = None
        if image_path.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.zeros(3, 224, 224)
        return image, torch.tensor(target, dtype=torch.float32)


class NutritionRegressor(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.backbone(x)


def build_dataloader(image_paths, targets, batch_size=8, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = FoodImageDataset(image_paths, targets, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def load_training_samples(input_dir, labels_csv=None, target_columns=None):
    image_paths = []
    targets = []
    target_columns = target_columns or DEFAULT_TARGET_COLUMNS

    if labels_csv:
        df = pd.read_csv(labels_csv)
        missing = [col for col in ["image_path"] + target_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Labels CSV is missing columns: {missing}")
        for _, row in df.iterrows():
            image_path = row["image_path"]
            if not os.path.isabs(image_path):
                image_path = str(Path(input_dir) / image_path)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image_paths.append(image_path)
            targets.append([float(row[col]) for col in target_columns])
    else:
        for image_file in sorted(Path(input_dir).glob("*")):
            if image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            image_paths.append(str(image_file))
            targets.append([0.0 for _ in target_columns])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return image_paths, targets, target_columns


def train_model(input_dir, output_dir, epochs=3, batch_size=8, labels_csv=None, target_columns=None):
    image_paths, targets, target_columns = load_training_samples(
        input_dir=input_dir,
        labels_csv=labels_csv,
        target_columns=target_columns,
    )

    loader = build_dataloader(image_paths, targets, batch_size=batch_size)
    model = NutritionRegressor(num_outputs=len(target_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, target in loader:
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss={running_loss / max(1, len(loader)):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "nutrition_model.pth"))
    print(f"Saved model to {output_dir}/nutrition_model.pth")


def predict_with_model(model_path, image_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = NutritionRegressor(num_outputs=len(DEFAULT_TARGET_COLUMNS))
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
    return preds.squeeze(0).cpu().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--labels_csv")
    parser.add_argument("--target_columns", nargs="+", default=None)
    args = parser.parse_args()
    train_model(
        args.input_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        labels_csv=args.labels_csv,
        target_columns=args.target_columns,
    )
