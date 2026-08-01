import cv2
import numpy as np

from fusion import DepthMaskFusion


depth = cv2.imread(
    "outputs/depth/depth_gray.png",
    cv2.IMREAD_GRAYSCALE
).astype(np.float32)

depth /= 255.0

mask = np.zeros(depth.shape, dtype=np.uint8)

h, w = depth.shape

cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)

fusion = DepthMaskFusion()

fused = fusion.fuse(mask, depth)

stats = fusion.statistics(fused)

cv2.imwrite("outputs/fusion/mask.png", mask)
cv2.imwrite("outputs/fusion/fused_depth.png",
            (fused*255).astype(np.uint8))

print("\nDepth Statistics")
print("-------------------------")

for k, v in stats.items():
    print(f"{k:8}: {v}")