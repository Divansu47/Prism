import numpy as np


class VolumeEstimator:

    def __init__(self,
                 pixel_area=1.0,
                 scale_factor=0.05):

        self.pixel_area = pixel_area
        self.scale_factor = scale_factor

    def estimate(self, fused_depth):

        valid = fused_depth[fused_depth > 0]

        if valid.size == 0:
            return 0.0

        volume = np.sum(valid)

        volume *= self.pixel_area

        volume *= self.scale_factor

        return float(volume)