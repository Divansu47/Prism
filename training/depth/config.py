from pathlib import Path

ROOT = Path(__file__).resolve().parent

WEIGHTS = ROOT / "weights" / "depth_anything_v2_vits.pth"

DEVICE = "cuda"

ENCODER = "vits"

INPUT_SIZE = 518