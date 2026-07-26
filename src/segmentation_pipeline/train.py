import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .dataset import FoodSegmentationDataset
    from .model import FoodSegmentationModel
except ImportError:  # pragma: no cover - allows running the file as a script
    from dataset import FoodSegmentationDataset
    from model import FoodSegmentationModel


def train_model(image_dir, mask_dir, labels_csv, output_dir, epochs=3, batch_size=2):
    if batch_size < 2:
        raise ValueError("Segmentation training requires batch_size >= 2 because the model uses batch normalization layers.")

    if labels_csv is None:
        labels_csv = os.path.join(Path(image_dir).parent, "labels_template.csv")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(
                "No labels CSV provided and no labels_template.csv found. "
                "Create one under the training data folder or pass --labels_csv explicitly."
            )

    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = FoodSegmentationDataset(image_dir, mask_dir, labels_csv, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_classes = len(dataset.class_names)
    model = FoodSegmentationModel(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss={running_loss / max(1, len(loader)):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "segmentation_model.pth"))
    print(f"Saved model to {output_dir}/segmentation_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--labels_csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    train_model(args.image_dir, args.mask_dir, args.labels_csv, args.output_dir, args.epochs, args.batch_size)
