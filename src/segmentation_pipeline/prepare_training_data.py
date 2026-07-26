from pathlib import Path
import shutil


def copy_starter_dataset(target_dir: str):
    """Copy the starter segmentation data into a target directory."""
    source = Path(__file__).resolve().parents[1] / ".." / "data" / "segmentation_training"
    target = Path(target_dir)
    if not source.exists():
        raise FileNotFoundError(f"Starter dataset not found: {source}")
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")
    shutil.copytree(source, target)
    print(f"Created starter dataset at {target}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", required=True)
    args = parser.parse_args()
    copy_starter_dataset(args.target_dir)
