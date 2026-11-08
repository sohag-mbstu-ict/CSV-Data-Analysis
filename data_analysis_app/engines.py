"""Keep a few DuckDB engines in memory so each question does not reload the CSV."""

import threading
from collections import OrderedDict
from contextlib import contextmanager

from utils.query_engine import Engine

MAX_ENGINES = 4


class _Entry:
    def __init__(self, engine):
        self.engine = engine
        # One question at a time per dataset: DuckDB connections are not shared across threads safely.
        self.lock = threading.Lock()
        self.closed = False


_cache = OrderedDict()
_cache_lock = threading.Lock()


def _close_entry(entry):
    with entry.lock:
        entry.closed = True
        entry.engine.close()


@contextmanager
def using(dataset):
    """Lock a dataset's engine for one question and yield it.

    Another request can evict and close an entry between get() and taking its lock,
    so a closed entry is skipped and the dataset is loaded again.
    """
    while True:
        entry = get(dataset)
        entry.lock.acquire()
        if not entry.closed:
            break
        entry.lock.release()
    try:
        yield entry.engine
    finally:
        entry.lock.release()


def put(dataset_id, engine):
    evicted = []
    with _cache_lock:
        previous = _cache.pop(dataset_id, None)
        if previous is not None:
            evicted.append(previous)
        _cache[dataset_id] = _Entry(engine)
        _cache.move_to_end(dataset_id)
        while len(_cache) > MAX_ENGINES:
            _, entry = _cache.popitem(last=False)
            evicted.append(entry)
    for entry in evicted:
        _close_entry(entry)


def get(dataset):
    """Return the cached entry for a dataset, loading its CSV on first use."""
    with _cache_lock:
        entry = _cache.get(dataset.id)
        if entry is not None:
            _cache.move_to_end(dataset.id)
            return entry
    engine = Engine(dataset.file.path)
    evicted = []
    with _cache_lock:
        entry = _cache.get(dataset.id)
        if entry is None:
            entry = _Entry(engine)
            _cache[dataset.id] = entry
            while len(_cache) > MAX_ENGINES:
                _, removed = _cache.popitem(last=False)
                evicted.append(removed)
        else:
            engine.close()
        _cache.move_to_end(dataset.id)
    for removed in evicted:
        _close_entry(removed)
    return entry


def discard(dataset_id):
    with _cache_lock:
        entry = _cache.pop(dataset_id, None)
    if entry is not None:
        _close_entry(entry)
