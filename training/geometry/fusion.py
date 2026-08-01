import cv2
import numpy as np


class DepthMaskFusion:

    def __init__(self):
        pass

    def fuse(self, mask, depth):

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = (mask > 0).astype(np.uint8)

        fused = depth.copy()

        fused[mask == 0] = 0

        return fused

    def statistics(self, fused):

        pixels = fused[fused > 0]

        if len(pixels) == 0:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "pixels": 0
            }

        return {
            "min": float(np.min(pixels)),
            "max": float(np.max(pixels)),
            "mean": float(np.mean(pixels)),
            "pixels": int(len(pixels))
        }