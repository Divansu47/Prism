import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

print("Loading Depth Anything V2...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def get_depth(image):
    """
    image : BGR numpy array (cv2.imread)

    returns:
        depth_map (float32 numpy array)
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    inputs = processor(images=pil, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

        predicted_depth = outputs.predicted_depth

        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = predicted_depth.cpu().numpy().astype(np.float32)

    depth -= depth.min()
    depth /= (depth.max() + 1e-8)

    return depth


if __name__ == "__main__":

    import sys
    import os

    img = cv2.imread(sys.argv[1])

    depth = get_depth(img)

    os.makedirs("outputs", exist_ok=True)

    np.save("outputs/depth.npy", depth)

    cv2.imwrite(
        "outputs/depth.png",
        (depth * 255).astype(np.uint8),
    )

    print("Saved outputs/depth.npy")
    print("Saved outputs/depth.png")