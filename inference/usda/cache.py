import threading
from collections import OrderedDict


DEFAULT_MAX_SIZE = 2048

_lock = threading.Lock()
_cache = OrderedDict()
_max_size = DEFAULT_MAX_SIZE


def configure(max_size):

    global _max_size

    with _lock:

        _max_size = max_size


def make_key(
    query,
    limit,
    prefer_generic,
    data_type,
):

    return (
        query.strip().lower(),
        limit,
        prefer_generic,
        data_type,
    )


def get(key):

    with _lock:

        if key not in _cache:
            return None

        _cache.move_to_end(key)

        return _cache[key]


def set(key, value):

    with _lock:

        _cache[key] = value

        _cache.move_to_end(key)

        while len(_cache) > _max_size:
            _cache.popitem(last=False)


def clear():

    with _lock:

        _cache.clear()


def size():

    with _lock:

        return len(_cache)


def cached_search(
    query,
    limit=10,
    prefer_generic=True,
    data_type=None,
):

    from inference.usda.search import search

    key = make_key(query, limit, prefer_generic, data_type)

    hit = get(key)

    if hit is not None:
        return hit

    result = search(
        query,
        limit=limit,
        prefer_generic=prefer_generic,
        data_type=data_type,
    )

    set(key, result)

    return result


def cached_search_best_match(
    query,
    prefer_generic=True,
    data_type=None,
):

    results = cached_search(
        query,
        limit=1,
        prefer_generic=prefer_generic,
        data_type=data_type,
    )

    if not results:
        return None

    return results[0]