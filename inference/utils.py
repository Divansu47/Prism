import cv2
import numpy as np


def smart_crop(image, margin=40):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image, (0, 0)

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    x = max(0, x - margin)
    y = max(0, y - margin)

    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    crop = image[y:y + h, x:x + w]

    return crop, (x, y)


def image_tiles(image, tile=640, overlap=80):

    H, W = image.shape[:2]

    stride = tile - overlap

    tiles = []

    for y in range(0, H, stride):

        for x in range(0, W, stride):

            y2 = min(y + tile, H)
            x2 = min(x + tile, W)

            patch = image[y:y2, x:x2]

            tiles.append(
                {
                    "image": patch,
                    "x": x,
                    "y": y,
                }
            )

    return tiles


def shift_boxes(result, dx, dy):

    if result.boxes is None:
        return result

    boxes = result.boxes.xyxy.cpu().numpy()

    boxes[:, [0, 2]] += dx
    boxes[:, [1, 3]] += dy

    result.boxes.xyxy[:] = boxes

    return result


def merge_results(results):

    if len(results) == 1:
        return results[0]

    best = results[0]

    for r in results[1:]:

        if r.boxes is None:
            continue

        if len(r.boxes) == 0:
            continue

        if best.boxes is None:

            best.boxes = r.boxes

            continue

        best.boxes = type(best.boxes).cat(
            [
                best.boxes,
                r.boxes,
            ]
        )

    return best


def resize_keep(image, size):

    h, w = image.shape[:2]

    scale = size / max(h, w)

    nh = int(h * scale)

    nw = int(w * scale)

    return cv2.resize(
        image,
        (nw, nh),
    )