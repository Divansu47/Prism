from pathlib import Path
from collections import Counter
import yaml


ROOT = Path(__file__).resolve().parents[1]

DATASET_YAML = ROOT / "training" / "vision" / "foodseg.yaml"


def load_class_names():

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config["names"], Path(config["path"])


def find_label_dirs(dataset_root):

    candidates = [
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]

    resolved = []

    for c in candidates:

        if c.exists():
            resolved.append(c)
        else:
            alt = ROOT / c

            if alt.exists():
                resolved.append(alt)

    return resolved


def count_instances(label_dir):

    counts = Counter()

    files = list(label_dir.glob("*.txt"))

    for file in files:

        with open(file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if not parts:
                    continue

                class_id = int(parts[0])

                counts[class_id] += 1

    return counts, len(files)


def count_images_per_class(label_dir):

    image_counts = Counter()

    files = list(label_dir.glob("*.txt"))

    for file in files:

        seen_classes = set()

        with open(file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if not parts:
                    continue

                class_id = int(parts[0])

                seen_classes.add(class_id)

        for class_id in seen_classes:
            image_counts[class_id] += 1

    return image_counts


def main():

    print("=" * 70)
    print("Class Distribution Analysis")
    print("=" * 70)
    print()

    names, dataset_root = load_class_names()

    label_dirs = find_label_dirs(dataset_root)

    if not label_dirs:
        print(f"No label directories found under: {dataset_root}")
        print("Checked: labels/train, labels/val")
        return

    total_instance_counts = Counter()
    total_image_counts = Counter()
    total_files = 0

    for label_dir in label_dirs:

        print(f"Scanning: {label_dir}")

        instance_counts, file_count = count_instances(label_dir)

        image_counts = count_images_per_class(label_dir)

        total_files += file_count

        total_instance_counts.update(instance_counts)

        total_image_counts.update(image_counts)

    print(f"\nTotal label files scanned: {total_files}\n")

    rows = []

    for class_id, class_name in names.items():

        instance_count = total_instance_counts.get(class_id, 0)

        image_count = total_image_counts.get(class_id, 0)

        rows.append((class_id, class_name, instance_count, image_count))

    rows.sort(key=lambda r: r[2])

    print(f"{'ID':<5}{'Class':<25}{'Instances':<12}{'Images':<10}")

    print("-" * 55)

    for class_id, class_name, instance_count, image_count in rows:

        print(f"{class_id:<5}{class_name:<25}{instance_count:<12}{image_count:<10}")

    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    zero_instance = [r for r in rows if r[2] == 0]

    low_instance = [r for r in rows if 0 < r[2] < 20]

    print(f"\nClasses with 0 instances       : {len(zero_instance)}")

    if zero_instance:
        for r in zero_instance:
            print(f"    - {r[1]}")

    print(f"\nClasses with <20 instances      : {len(low_instance)}")

    if low_instance:
        for r in low_instance:
            print(f"    - {r[1]} ({r[2]} instances)")

    print()

    pizza_row = next((r for r in rows if r[1] == "pizza"), None)

    chicken_row = next((r for r in rows if r[1] == "chicken duck"), None)

    if pizza_row:
        print(f"pizza          : {pizza_row[2]} instances, {pizza_row[3]} images")

    if chicken_row:
        print(f"chicken duck   : {chicken_row[2]} instances, {chicken_row[3]} images")

    print()

    print("=" * 70)


if __name__ == "__main__":
    main()