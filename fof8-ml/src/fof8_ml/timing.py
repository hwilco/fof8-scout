from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from time import perf_counter


def timing_enabled(cfg: object) -> bool:
    """Return whether timing diagnostics are enabled for a config-like object."""

    diagnostics = _cfg_get(cfg, "diagnostics", {})
    timing = _cfg_get(diagnostics, "timing", {})
    return bool(_cfg_get(timing, "enabled", False))


def _cfg_get(cfg: object, key: str, default: object = None) -> object:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


@contextmanager
def timed_step(label: str, *, enabled: bool) -> Iterator[None]:
    """Print start/end wall-clock timing for a named step when enabled."""

    if not enabled:
        yield
        return

    start = perf_counter()
    print(f"[timing] START {label}")
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        print(f"[timing] END {label} elapsed={elapsed:.3f}s")
