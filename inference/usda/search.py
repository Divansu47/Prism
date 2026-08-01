import re
from difflib import SequenceMatcher

from inference.usda.loader import get_connection


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "with",
    "of",
    "net",
    "wt",
    "weight",
    "fl",
    "oz",
    "ml",
    "energ",
    "drink",
    "body",
    # Add common generic food/packaging terms below
    "original",
    "flavor",
    "classic",
    "regular",
    "brand",
    # Packaging Types
    "bottle",
    "can",
    "pack",
    "package",
    "carton",
    "pouch",
    "jar",
    "tub",
    "tin",
    "packet",
    "wrapper",
    "box",
    "bag",
    "jug",
    "case",
    "container",
    # Materials
    "plastic",
    "glass",
    "paper",
    "foil",
    "cardboard",
    "aluminum",
    # Quantities & Measurements
    "g",
    "kg",
    "lb",
    "lbs",
    "ct",
    "count",
    "pcs",
    "pieces",
    "size",
    "serving",
    "servings",
    "litre",
    "liter",
    "liters",
    "pint",
    "pints",
    "quart",
    "quarts",
    "gallon",
    # Common Food Attributes
    "food",
    "snack",
    "beverage",
    "juice",
    "water",
    "soda",
    "sauce",
    "syrup",
    "mix",
    "powder",
    "liquid",
    "paste",
    "puree",
    "organic",
    "natural",
    "sweet",
    "salted",
    "spicy",
    "hot",
    "mild",
    "fresh",
    "frozen",
    "dried",
    "baked",
    "fried",
    "roasted",
    "premium",
    "gourmet",
    "diet",
    "light",
    "lite",
    "zero",
    "sugar",
    "free",
    "low",
    "high",
    "rich",
    "creamy",
    "chunky",
    "smooth",
    "whole",
    "sliced",
    "diced",
}

MIN_TOKEN_LENGTH = 4
MIN_OVERLAP_RATIO = 0.34
MIN_QUERY_TOKENS_FOR_OVERLAP = 3

FUZZY_MIN_RATIO = 0.78
FUZZY_CANDIDATE_POOL = 15000


def normalize_query(text):

    text = str(text).strip().lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def _tokenize(text, min_length=2):

    words = text.split()

    tokens = [
        w for w in words
        if w not in STOPWORDS and len(w) >= min_length
    ]

    return tokens


def search(
    query,
    limit=10,
    prefer_generic=True,
    data_type=None,
):

    normalized = normalize_query(query)

    if not normalized:
        return []

    results = _strict_search(
        normalized,
        limit=limit,
        prefer_generic=prefer_generic,
        data_type=data_type,
    )

    if results:
        return results

    results = _best_token_search(
        normalized,
        limit=limit,
        prefer_generic=prefer_generic,
        data_type=data_type,
    )

    if results:
        return results

    results = _fuzzy_search(
        normalized,
        limit=limit,
        prefer_generic=prefer_generic,
        data_type=data_type,
    )

    if results:
        return results

    return _token_overlap_search(
        normalized,
        limit=limit,
        data_type=data_type,
    )


def _strict_search(
    normalized,
    limit,
    prefer_generic,
    data_type,
):

    conn = get_connection()

    prefix_pattern = f"{normalized}%"
    contains_pattern = f"%{normalized}%"

    sql = """
        SELECT
            *,
            CASE
                WHEN normalized_name = ? THEN 0
                WHEN normalized_name LIKE ? THEN 1
                ELSE 2
            END AS match_rank
        FROM foods
        WHERE (
            normalized_name = ?
            OR normalized_name LIKE ?
            OR normalized_name LIKE ?
        )
    """

    params = [
        normalized,
        prefix_pattern,
        normalized,
        prefix_pattern,
        contains_pattern,
    ]

    if data_type is not None:

        sql += " AND data_type = ?"

        params.append(data_type)

    order_clause = " ORDER BY match_rank ASC, "

    if prefer_generic:

        order_clause += "is_generic DESC, "

    order_clause += "score ASC, publication_date DESC"

    sql += order_clause

    sql += " LIMIT ?"

    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

    return _rows_to_records(rows)


def _best_token_search(
    normalized,
    limit,
    prefer_generic,
    data_type,
):

    tokens = _tokenize(normalized, min_length=MIN_TOKEN_LENGTH)

    tokens = sorted(tokens, key=len, reverse=True)

    conn = get_connection()

    for token in tokens:

        results = _token_substring_search(
            conn,
            token,
            limit,
            prefer_generic,
            data_type,
        )

        if results:

            for record in results:
                record["matched_token"] = token

            return results

        results = _token_substring_search_ignore_spaces(
            conn,
            token,
            limit,
            prefer_generic,
            data_type,
        )

        if results:

            for record in results:
                record["matched_token"] = token
                record["matched_via"] = "space_insensitive"

            return results

    return []


def _token_substring_search(
    conn,
    token,
    limit,
    prefer_generic,
    data_type,
):

    contains_pattern = f"%{token}%"

    sql = """
        SELECT *
        FROM foods
        WHERE normalized_name LIKE ?
    """

    params = [contains_pattern]

    if data_type is not None:

        sql += " AND data_type = ?"

        params.append(data_type)

    order_clause = " ORDER BY "

    if prefer_generic:

        order_clause += "is_generic DESC, "

    order_clause += "score ASC, publication_date DESC"

    sql += order_clause

    sql += " LIMIT ?"

    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

    return _rows_to_records(rows)


def _token_substring_search_ignore_spaces(
    conn,
    token,
    limit,
    prefer_generic,
    data_type,
):

    contains_pattern = f"%{token}%"

    sql = """
        SELECT *
        FROM foods
        WHERE REPLACE(normalized_name, ' ', '') LIKE ?
    """

    params = [contains_pattern]

    if data_type is not None:

        sql += " AND data_type = ?"

        params.append(data_type)

    order_clause = " ORDER BY "

    if prefer_generic:

        order_clause += "is_generic DESC, "

    order_clause += "score ASC, publication_date DESC"

    sql += order_clause

    sql += " LIMIT ?"

    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

    return _rows_to_records(rows)


def _fuzzy_search(
    normalized,
    limit,
    prefer_generic,
    data_type,
):

    # Catches single-character OCR misreads (e.g. "mwnster" vs
    # "monster") that substring matching can never catch, since
    # the character sequence itself differs. Candidate pool is
    # narrowed by first letter to keep this affordable; similarity
    # is computed per-word in Python via edit-distance ratio.

    tokens = _tokenize(normalized, min_length=MIN_TOKEN_LENGTH)

    tokens = sorted(tokens, key=len, reverse=True)

    if not tokens:
        return []

    conn = get_connection()

    for token in tokens:

        first_char = token[0]

        prefix_pattern = f"{first_char}%"

        sql = """
            SELECT *
            FROM foods
            WHERE normalized_name LIKE ?
        """

        params = [prefix_pattern]

        if data_type is not None:

            sql += " AND data_type = ?"

            params.append(data_type)

        # Add the ORDER BY clause here, after all WHERE conditions are set
        sql += " ORDER BY LENGTH(normalized_name) ASC"

        sql += " LIMIT ?"

        params.append(FUZZY_CANDIDATE_POOL)

        rows = conn.execute(sql, params).fetchall()

        scored = []

        for row in rows:

            record = dict(row)

            words = record["normalized_name"].split()

            best_ratio = 0.0

            for word in words:

                ratio = SequenceMatcher(None, token, word).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio

            if best_ratio >= FUZZY_MIN_RATIO:

                record["match_fuzzy_ratio"] = best_ratio
                record["matched_token"] = token
                record["matched_via"] = "fuzzy"

                scored.append((best_ratio, record))

        if not scored:
            continue

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1]["score"],
            )
        )

        results = []
        seen = set()

        for ratio, record in scored:

            fdc_id = record["fdc_id"]

            if fdc_id in seen:
                continue

            seen.add(fdc_id)

            results.append(record)

            if len(results) >= limit:
                break

        if results:
            return results

    return []


def _token_overlap_search(
    normalized,
    limit,
    data_type,
    candidate_pool_size=500,
):

    query_tokens = set(_tokenize(normalized, min_length=MIN_TOKEN_LENGTH))

    if len(query_tokens) < MIN_QUERY_TOKENS_FOR_OVERLAP:
        return []

    conn = get_connection()

    like_clauses = []
    params = []

    for token in query_tokens:

        like_clauses.append("normalized_name LIKE ?")

        params.append(f"%{token}%")

    sql = f"""
        SELECT *
        FROM foods
        WHERE ({" OR ".join(like_clauses)})
    """

    if data_type is not None:

        sql += " AND data_type = ?"

        params.append(data_type)

    sql += " LIMIT ?"

    params.append(candidate_pool_size)

    rows = conn.execute(sql, params).fetchall()

    scored = []

    for row in rows:

        record = dict(row)

        candidate_tokens = set(_tokenize(record["normalized_name"], min_length=MIN_TOKEN_LENGTH))

        if not candidate_tokens:
            continue

        overlap = query_tokens & candidate_tokens

        if not overlap:
            continue

        overlap_ratio = len(overlap) / len(query_tokens)

        if overlap_ratio < MIN_OVERLAP_RATIO:
            continue

        record["match_overlap_ratio"] = overlap_ratio

        scored.append((overlap_ratio, record))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1]["score"],
        )
    )

    results = []

    seen = set()

    for overlap_ratio, record in scored:

        fdc_id = record["fdc_id"]

        if fdc_id in seen:
            continue

        seen.add(fdc_id)

        results.append(record)

        if len(results) >= limit:
            break

    return results


def _rows_to_records(rows):

    results = []

    seen = set()

    for row in rows:

        fdc_id = row["fdc_id"]

        if fdc_id in seen:
            continue

        seen.add(fdc_id)

        record = dict(row)

        record.pop("match_rank", None)

        results.append(record)

    return results


def search_best_match(
    query,
    prefer_generic=True,
    data_type=None,
):

    results = search(
        query,
        limit=1,
        prefer_generic=prefer_generic,
        data_type=data_type,
    )

    if not results:
        return None

    return results[0]