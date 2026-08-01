import threading


_lock = threading.Lock()
_classifier = None

MODEL_NAME = "nateraw/food"

CONFIDENCE_THRESHOLD = 0.60


def _load_classifier():

    global _classifier

    if _classifier is not None:
        return _classifier

    with _lock:

        if _classifier is None:

            from transformers import pipeline

            print("\nLoading dish-level classifier (Food-101)...")

            _classifier = pipeline(
                "image-classification",
                model=MODEL_NAME,
            )

    return _classifier


def classify_dish(image_bgr):

    try:
        classifier = _load_classifier()
    except Exception as e:
        print(f"Dish classifier unavailable: {e}")
        return None

    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    try:
        results = classifier(pil_image)
    except Exception as e:
        print(f"Dish classification failed: {e}")
        return None

    if not results:
        return None

    top = results[0]

    label = str(top["label"]).replace("_", " ").lower()
    score = float(top["score"])

    return {
        "label": label,
        "confidence": score,
        "is_confident": score >= CONFIDENCE_THRESHOLD,
    }