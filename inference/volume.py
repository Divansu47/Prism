import cv2
import numpy as np


def estimate_volume(result, depth):

    foods = []

    if result.masks is None:
        return foods

    if hasattr(result, "refined_masks"):
        masks = result.refined_masks
    else:
        masks = result.masks.data.cpu().numpy()
    boxes = result.boxes

    for i in range(len(boxes)):

        mask = masks[i]

        # Resize mask to match depth map resolution
        if mask.shape != depth.shape:

            mask = cv2.resize(
                mask.astype(np.float32),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask = mask > 0.5

        pixel_area = int(mask.sum())

        if pixel_area == 0:
            continue

        avg_depth = float(depth[mask].mean())

        foods.append(
            {
                "class": result.names[int(boxes.cls[i])],
                "confidence": float(boxes.conf[i]),
                "pixel_area": pixel_area,
                "avg_depth": avg_depth,
                "relative_volume": pixel_area * avg_depth,
            }
        )

    return foods