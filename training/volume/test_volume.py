from pathlib import Path
import cv2
import numpy as np

from estimator import VolumeEstimator


ROOT = Path(__file__).resolve().parents[2]

fusion_path = ROOT / "outputs" / "fusion" / "fused_depth.png"

depth = cv2.imread(
    str(fusion_path),
    cv2.IMREAD_GRAYSCALE
)

if depth is None:
    raise FileNotFoundError(fusion_path)

depth = depth.astype(np.float32) / 255.0

estimator = VolumeEstimator()

volume = estimator.estimate(depth)

print()

print("=" * 40)
print("Estimated Relative Volume")
print("=" * 40)
print(f"{volume:.2f} cubic units")
print("=" * 40)