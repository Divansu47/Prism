import cv2
import numpy as np


def save_depth(depth, output_path):

    img = (depth * 255).astype(np.uint8)

    cv2.imwrite(output_path, img)


def apply_colormap(depth):

    img = (depth * 255).astype(np.uint8)

    img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

    return img