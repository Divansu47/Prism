from pathlib import Path
import cv2

from models.depth_anything import DepthAnythingV2
from utils.image import save_depth, apply_colormap


IMAGE_PATH = "assets/test.jpg"

OUTPUT_DIR = Path("outputs/depth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    model = DepthAnythingV2()

    depth = model.predict(IMAGE_PATH)

    save_depth(depth, OUTPUT_DIR / "depth_gray.png")

    colored = apply_colormap(depth)

    cv2.imwrite(str(OUTPUT_DIR / "depth_color.png"), colored)

    print("Done.")
    print("Saved:")
    print(OUTPUT_DIR / "depth_gray.png")
    print(OUTPUT_DIR / "depth_color.png")


if __name__ == "__main__":
    main()