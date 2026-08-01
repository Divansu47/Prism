from pathlib import Path
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from training.vision.converters.base_converter import BaseConverter


class FoodSeg103Converter(BaseConverter):

    def __init__(self, dataset_root: str, output_root: str):

        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

        self.image_root = self.dataset_root / "Images" / "img_dir"
        self.mask_root = self.dataset_root / "Images" / "ann_dir"

        self.image_output = self.output_root / "images"
        self.label_output = self.output_root / "labels"

        self._create_directories()

    def _create_directories(self):

        for split in ["train", "val"]:

            (self.image_output / split).mkdir(parents=True, exist_ok=True)
            (self.label_output / split).mkdir(parents=True, exist_ok=True)

    def convert(self):

        self._convert_split("train", "train")
        self._convert_split("test", "val")

        print("\nDataset conversion completed.")

    def _convert_split(self, input_split: str, output_split: str):

        image_dir = self.image_root / input_split
        mask_dir = self.mask_root / input_split

        image_files = sorted(image_dir.glob("*.jpg"))

        print(f"\nConverting {input_split} ({len(image_files)} images)")

        for image_path in tqdm(image_files):

            mask_path = mask_dir / f"{image_path.stem}.png"

            if not mask_path.exists():
                continue

            self._process_image(
                image_path,
                mask_path,
                output_split
            )

    def _process_image(
        self,
        image_path: Path,
        mask_path: Path,
        split: str
    ):

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None or mask is None:
            return

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        h, w = mask.shape

        label_path = self.label_output / split / f"{image_path.stem}.txt"

        with open(label_path, "w") as file:

            class_ids = np.unique(mask)

            for class_id in class_ids:

                if class_id == 0:
                    continue

                binary_mask = np.uint8(mask == class_id)

                contours, _ = cv2.findContours(
                    binary_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:

                    if len(contour) < 3:
                        continue

                    epsilon = 0.002 * cv2.arcLength(
                        contour,
                        True
                    )

                    contour = cv2.approxPolyDP(
                        contour,
                        epsilon,
                        True
                    )

                    if len(contour) < 3:
                        continue

                    polygon = []

                    for point in contour:

                        x, y = point[0]

                        polygon.append(x / w)
                        polygon.append(y / h)

                    polygon_string = " ".join(
                        f"{value:.6f}" for value in polygon
                    )

                    file.write(
                        f"{int(class_id)-1} {polygon_string}\n"
                    )

        shutil.copy2(
            image_path,
            self.image_output / split / image_path.name
        )


if __name__ == "__main__":

    converter = FoodSeg103Converter(

        dataset_root="datasets/foodseg103/FoodSeg103",
        output_root="datasets/foodseg103_yolo"
    )

    converter.convert()