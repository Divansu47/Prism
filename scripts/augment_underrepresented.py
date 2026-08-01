from pathlib import Path
from collections import Counter
import random

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]

DATASET_YAML = ROOT / "training" / "vision" / "foodseg.yaml"

RANDOM_SEED = 42

# Tiered augmentation multipliers based on current TRAIN-split
# instance count. Only train images are touched; val is never
# augmented to avoid leakage.
TIER_THRESHOLDS = [
    (20, 8),
    (50, 5),
    (150, 3),
]

DEFAULT_MULTIPLIER = 0

ROTATION_RANGE_DEG = (-15, 15)
SCALE_RANGE = (0.9, 1.15)
BRIGHTNESS_RANGE = (-25, 25)
CONTRAST_RANGE = (0.85, 1.15)
FLIP_PROBABILITY = 0.5


def load_config():

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = ROOT / config["path"]

    return config["names"], dataset_root


def get_multiplier(instance_count):

    for threshold, multiplier in TIER_THRESHOLDS:

        if instance_count < threshold:
            return multiplier

    return DEFAULT_MULTIPLIER


def count_train_instances(label_dir):

    counts = Counter()

    for file in label_dir.glob("*.txt"):

        with open(file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                class_id = int(line.split()[0])

                counts[class_id] += 1

    return counts


def find_target_images(label_dir, target_class_ids):

    matches = []

    for file in label_dir.glob("*.txt"):

        with open(file, "r", encoding="utf-8") as f:

            lines = f.readlines()

        classes_in_file = set()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])

            classes_in_file.add(class_id)

        hit_classes = classes_in_file & target_class_ids

        if hit_classes:
            matches.append((file, hit_classes))

    return matches


def parse_label_file(label_path):

    instances = []

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            class_id = int(parts[0])

            coords = [float(x) for x in parts[1:]]

            points = []

            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i + 1]))

            instances.append((class_id, points))

    return instances


def write_label_file(label_path, instances):

    lines = []

    for class_id, points in instances:

        flat = []

        for x, y in points:
            flat.append(f"{x:.6f}")
            flat.append(f"{y:.6f}")

        lines.append(str(class_id) + " " + " ".join(flat))

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def denormalize(points, width, height):

    return [(x * width, y * height) for x, y in points]


def normalize(points, width, height):

    return [(x / width, y / height) for x, y in points]


def build_affine_matrix(center, angle_deg, scale):

    return cv2.getRotationMatrix2D(center, angle_deg, scale)


def transform_points(points, matrix):

    transformed = []

    for x, y in points:

        new_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
        new_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]

        transformed.append((new_x, new_y))

    return transformed


def clip_points(points, width, height):

    clipped = []

    for x, y in points:

        cx = min(max(x, 0.0), width - 1.0)
        cy = min(max(y, 0.0), height - 1.0)

        clipped.append((cx, cy))

    return clipped


def polygon_area(points):

    if len(points) < 3:
        return 0.0

    area = 0.0

    n = len(points)

    for i in range(n):

        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]

        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def augment_once(image, instances, apply_flip, angle_deg, scale, brightness, contrast):

    height, width = image.shape[:2]

    work_image = image.copy()

    work_instances = []

    for class_id, norm_points in instances:

        px_points = denormalize(norm_points, width, height)

        work_instances.append((class_id, px_points))

    if apply_flip:

        work_image = cv2.flip(work_image, 1)

        flipped_instances = []

        for class_id, px_points in work_instances:

            flipped_points = [(width - x, y) for x, y in px_points]

            flipped_instances.append((class_id, flipped_points))

        work_instances = flipped_instances

    center = (width / 2.0, height / 2.0)

    matrix = build_affine_matrix(center, angle_deg, scale)

    work_image = cv2.warpAffine(
        work_image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT_101,
    )

    rotated_instances = []

    for class_id, px_points in work_instances:

        new_points = transform_points(px_points, matrix)

        new_points = clip_points(new_points, width, height)

        if polygon_area(new_points) < 4.0:
            continue

        rotated_instances.append((class_id, new_points))

    work_image = cv2.convertScaleAbs(
        work_image,
        alpha=contrast,
        beta=brightness,
    )

    final_instances = []

    for class_id, px_points in rotated_instances:

        norm_points = normalize(px_points, width, height)

        final_instances.append((class_id, norm_points))

    return work_image, final_instances


def main():

    random.seed(RANDOM_SEED)

    print("=" * 70)
    print("Underrepresented Class Augmentation")
    print("=" * 70)
    print()

    names, dataset_root = load_config()

    train_images_dir = dataset_root / "images" / "train"
    train_labels_dir = dataset_root / "labels" / "train"

    if not train_images_dir.exists() or not train_labels_dir.exists():
        print(f"Could not find train images/labels under: {dataset_root}")
        return

    print("Counting current train instance distribution...")

    instance_counts = count_train_instances(train_labels_dir)

    target_class_ids = set()
    multiplier_by_class = {}

    print()
    print(f"{'ID':<5}{'Class':<25}{'Instances':<12}{'Multiplier':<10}")
    print("-" * 55)

    for class_id, class_name in names.items():

        count = instance_counts.get(class_id, 0)

        multiplier = get_multiplier(count)

        if multiplier > 0:

            target_class_ids.add(class_id)

            multiplier_by_class[class_id] = multiplier

            print(f"{class_id:<5}{class_name:<25}{count:<12}{multiplier:<10}")

    if not target_class_ids:
        print("\nNo underrepresented classes found under current thresholds.")
        return

    print()

    print("Scanning train label files for target classes...")

    matches = find_target_images(train_labels_dir, target_class_ids)

    print(f"Found {len(matches)} source images containing target classes.\n")

    total_generated = 0

    for label_path, hit_classes in matches:

        image_stem = label_path.stem

        image_path = None

        for ext in (".jpg", ".jpeg", ".png"):

            candidate = train_images_dir / f"{image_stem}{ext}"

            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"Warning: no image found for label {label_path.name}, skipping.")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: could not read image {image_path.name}, skipping.")
            continue

        instances = parse_label_file(label_path)

        n_augmentations = max(multiplier_by_class[c] for c in hit_classes)

        for aug_index in range(n_augmentations):

            apply_flip = random.random() < FLIP_PROBABILITY

            angle_deg = random.uniform(*ROTATION_RANGE_DEG)

            scale = random.uniform(*SCALE_RANGE)

            brightness = random.uniform(*BRIGHTNESS_RANGE)

            contrast = random.uniform(*CONTRAST_RANGE)

            aug_image, aug_instances = augment_once(
                image,
                instances,
                apply_flip,
                angle_deg,
                scale,
                brightness,
                contrast,
            )

            if not aug_instances:
                continue

            new_stem = f"{image_stem}_aug{aug_index}"

            new_image_path = train_images_dir / f"{new_stem}{image_path.suffix}"

            new_label_path = train_labels_dir / f"{new_stem}.txt"

            cv2.imwrite(str(new_image_path), aug_image)

            write_label_file(new_label_path, aug_instances)

            total_generated += 1

        if total_generated % 100 == 0 and total_generated > 0:
            print(f"Generated {total_generated} augmented samples so far...")

    print()
    print("=" * 70)
    print("Augmentation complete")
    print(f"Total new image/label pairs generated: {total_generated}")
    print("=" * 70)


if __name__ == "__main__":
    main()