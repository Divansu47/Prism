COVERAGE_THRESHOLD = 0.60
CONFIDENCE_THRESHOLD = 0.75


def check_dominant_detection(food, image_area):

    pixel_area = food.get("pixel_area", 0)
    confidence = food.get("confidence", 0.0)

    if image_area <= 0:
        return False, None

    coverage = pixel_area / image_area

    if coverage >= COVERAGE_THRESHOLD and confidence < CONFIDENCE_THRESHOLD:

        reason = (
            f"Detection covers {coverage * 100:.1f}% of frame "
            f"at only {confidence * 100:.1f}% confidence"
        )

        return True, reason

    return False, None


def check_dish_mismatch(dish_result, foods):

    if dish_result is None:
        return False, None

    if not dish_result.get("is_confident"):
        return False, None

    dish_label = dish_result["label"]

    detected_classes = {
        str(f.get("class", "")).lower() for f in foods
    }

    match_found = any(
        dish_label in cls or cls in dish_label
        for cls in detected_classes
    )

    if not match_found:

        reason = (
            f"Dish classifier says '{dish_label}' "
            f"({dish_result['confidence'] * 100:.1f}% confidence) "
            f"but segmentation did not detect a matching class"
        )

        return True, reason

    return False, None