from transformers import pipeline
import numpy as np
from PIL import Image


class DepthAnythingV2:

    def __init__(self,
                 model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 device="cuda"):

        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=device
        )

    def predict(self, image):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        result = self.pipe(image)

        depth = np.array(result["depth"], dtype=np.float32)

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth