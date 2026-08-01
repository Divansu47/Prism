from rapidfuzz import fuzz


def normalize(text: str):

    return (
        text.lower()
            .replace("_", " ")
            .replace("-", " ")
            .strip()
    )


def _candidate_set(query, search_index):

    query = normalize(query)

    candidates = set()

    if query in search_index:
        candidates.update(search_index[query])

    for word in query.split():

        if word in search_index:
            candidates.update(search_index[word])

    return candidates


def _score(query, food):

    query = normalize(query)
    food = normalize(food)

    score = fuzz.token_sort_ratio(query, food)

    q_words = set(query.split())
    f_words = set(food.split())

    # Exact match
    if query == food:
        score += 100

    # Whole word overlap
    score += 20 * len(q_words & f_words)

    # Prefer fewer extra words
    score -= 2 * abs(len(f_words) - len(q_words))

    return score


def best_match(query, database, search_index):

    candidates = _candidate_set(query, search_index)

    if not candidates:
        candidates = database.keys()

    best_food = None
    best_score = -1

    for food in candidates:

        score = _score(query, food)

        if score > best_score:
            best_score = score
            best_food = food

    if best_food is None:
        return None, 0, "None"

    if normalize(best_food) == normalize(query):
        method = "Exact"
    else:
        method = "Ranked"

    return best_food, round(best_score, 2), method