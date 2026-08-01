import re
import threading


_lock = threading.Lock()
_reader = None

MIN_TEXT_BLOCKS = 3
MIN_AVG_CONFIDENCE = 0.35
MIN_TEXT_AREA_RATIO = 0.03
MIN_TOKEN_CONFIDENCE = 0.20

WEIGHT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s?(g|kg|ml|l)\b",
    re.IGNORECASE,
)

MIN_PLAUSIBLE_WEIGHT_G = 5.0
MAX_PLAUSIBLE_WEIGHT_G = 5000.0

NOISE_WORDS = {
    "net",
    "wt",
    "weight",
    "ingredients",
    "nutrition",
    "facts",
    "serving",
    "size",
}


def _load_reader():

    global _reader

    if _reader is not None:
        return _reader

    with _lock:

        if _reader is None:

            import easyocr

            print("\nLoading OCR reader (EasyOCR)...")

            _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


def _polygon_area(points):

    if len(points) < 3:
        return 0.0

    area = 0.0

    n = len(points)

    for i in range(n):

        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]

        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def _extract_weight_grams(text):

    matches = WEIGHT_PATTERN.findall(text)

    for value_str, unit in matches:

        value = float(value_str)

        unit = unit.lower()

        if unit == "kg":
            grams = value * 1000.0
        elif unit == "l":
            grams = value * 1000.0
        elif unit == "ml":
            grams = value
        else:
            grams = value

        if MIN_PLAUSIBLE_WEIGHT_G <= grams <= MAX_PLAUSIBLE_WEIGHT_G:
            return grams

    return None


def _clean_query_text(text):

    words = re.findall(r"[a-zA-Z]+", text.lower())

    filtered = [w for w in words if w not in NOISE_WORDS and len(w) > 1]

    return " ".join(filtered)


def detect_package_and_text(image_bgr):

    try:
        reader = _load_reader()
    except Exception as e:

        print(f"OCR reader unavailable: {e}")

        return {
            "is_packaged": False,
            "raw_text": "",
            "query_text": "",
            "weight_grams": None,
        }

    import cv2

    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width = image_bgr.shape[:2]

    image_area = float(height * width)

    try:
        results = reader.readtext(rgb_image)
    except Exception as e:

        print(f"OCR read failed: {e}")

        return {
            "is_packaged": False,
            "raw_text": "",
            "query_text": "",
            "weight_grams": None,
        }

    if not results:

        return {
            "is_packaged": False,
            "raw_text": "",
            "query_text": "",
            "weight_grams": None,
        }

    block_count = len(results)

    confidences = [float(r[2]) for r in results]

    avg_confidence = sum(confidences) / len(confidences)

    total_text_area = 0.0

    for box, _, _ in results:

        total_text_area += _polygon_area(box)

    area_ratio = total_text_area / image_area if image_area > 0 else 0.0

    is_packaged = (
        block_count >= MIN_TEXT_BLOCKS
        and avg_confidence >= MIN_AVG_CONFIDENCE
    ) or area_ratio >= MIN_TEXT_AREA_RATIO

    kept_tokens = []

    for _, text, confidence in results:

        if confidence >= MIN_TOKEN_CONFIDENCE:
            kept_tokens.append(text)

    raw_text = " ".join(kept_tokens)

    query_text = _clean_query_text(raw_text)

    weight_grams = _extract_weight_grams(raw_text)

    print("\n========== OCR SCAN ==========")
    print(f"Text blocks detected : {block_count}")
    print(f"Average confidence   : {avg_confidence:.3f}")
    print(f"Text area ratio      : {area_ratio:.3f}")
    print(f"Is packaged product  : {is_packaged}")

    if raw_text:
        print(f"Detected text        : {raw_text}")

    if weight_grams is not None:
        print(f"Detected net weight   : {weight_grams:.1f} g")

    print("===============================\n")

    return {
        "is_packaged": is_packaged,
        "raw_text": raw_text,
        "query_text": query_text,
        "weight_grams": weight_grams,
    }