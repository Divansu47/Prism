import numpy as np


def estimate_volume(results, depth_map):
    foods = []

    result = results[0]

    if result.masks is None:
        return foods

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    names = result.names

    for mask, cls in zip(masks, classes):

        mask = mask > 0.5

        pixel_area = int(mask.sum())

        if pixel_area == 0:
            continue

        avg_depth = float(depth_map[mask].mean())

        relative_volume = pixel_area * avg_depth

        foods.append(
            {
                "class": names[cls],
                "pixel_area": pixel_area,
                "avg_depth": avg_depth,
                "relative_volume": relative_volume,
            }
        )

    return foods