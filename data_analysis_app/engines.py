"""Keep a few DuckDB engines in memory so each question does not reload the CSV."""

import threading
from collections import OrderedDict

from query_engine import Engine

MAX_ENGINES = 4


class _Entry:
    def __init__(self, engine):
        self.engine = engine
        # One question at a time per dataset: DuckDB connections are not shared across threads safely.
        self.lock = threading.Lock()


_cache = OrderedDict()
_cache_lock = threading.Lock()


def put(dataset_id, engine):
    with _cache_lock:
        _cache[dataset_id] = _Entry(engine)
        _cache.move_to_end(dataset_id)
        while len(_cache) > MAX_ENGINES:
            # Dropped engines close when no request is using them any more.
            _cache.popitem(last=False)


def get(dataset):
    """Return the cached entry for a dataset, loading its CSV on first use."""
    with _cache_lock:
        entry = _cache.get(dataset.id)
        if entry is not None:
            _cache.move_to_end(dataset.id)
            return entry
    engine = Engine(dataset.file.path)
    with _cache_lock:
        entry = _cache.get(dataset.id)
        if entry is None:
            entry = _Entry(engine)
            _cache[dataset.id] = entry
            while len(_cache) > MAX_ENGINES:
                _cache.popitem(last=False)
        else:
            engine.close()
        _cache.move_to_end(dataset.id)
        return entry


def discard(dataset_id):
    with _cache_lock:
        _cache.pop(dataset_id, None)
