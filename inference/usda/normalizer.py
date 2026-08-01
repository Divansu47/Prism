import re


STOPWORDS = {
    "raw",
    "cooked",
    "fresh",
    "frozen",
    "canned",
    "with",
    "without",
    "and",
    "or",
    "the",
    "a",
    "an",
}

PLURAL_EXCEPTIONS = {
    "beans": "bean",
    "peas": "pea",
    "greens": "green",
    "oats": "oat",
    "grapes": "grape",
    "berries": "berry",
    "noodles": "noodle",
    "chips": "chip",
}


def basic_normalize(text):

    text = str(text).strip().lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def singularize_word(word):

    if word in PLURAL_EXCEPTIONS:
        return PLURAL_EXCEPTIONS[word]

    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"

    if len(word) > 3 and word.endswith("es") and word[-3] not in "aeiou":
        return word[:-2]

    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def strip_stopwords(text):

    words = text.split()

    filtered = [w for w in words if w not in STOPWORDS]

    if not filtered:
        return text

    return " ".join(filtered)


def normalize_for_search(text):

    text = basic_normalize(text)

    text = strip_stopwords(text)

    words = text.split()

    words = [singularize_word(w) for w in words]

    return " ".join(words)


def normalize_for_storage(text):

    return basic_normalize(text)


def build_search_variants(text):

    variants = []

    base = basic_normalize(text)

    if base:
        variants.append(base)

    stripped = strip_stopwords(base)

    if stripped and stripped not in variants:
        variants.append(stripped)

    singular = normalize_for_search(text)

    if singular and singular not in variants:
        variants.append(singular)

    return variants