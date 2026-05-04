"""Target registry for config-driven resolution of named target builders."""

from collections.abc import Callable

import polars as pl

from fof8_core.loader import FOF8Loader

TargetBuilder = Callable[[FOF8Loader], pl.DataFrame]
TARGET_REGISTRY: dict[str, TargetBuilder] = {}
_BUILTINS_REGISTERED = False


def register_target(name: str) -> Callable[[TargetBuilder], TargetBuilder]:
    def decorator(func: TargetBuilder) -> TargetBuilder:
        TARGET_REGISTRY[name] = func
        return func

    return decorator


def _register_builtin_targets() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from fof8_core.targets.approximate_value import get_career_value_metrics
    from fof8_core.targets.awards import get_awards
    from fof8_core.targets.career import get_career_outcomes
    from fof8_core.targets.economic import get_economic_targets
    from fof8_core.targets.financial import (
        get_annual_financials,
        get_merit_cap_share,
        get_peak_overall,
    )

    TARGET_REGISTRY["career_outcomes"] = get_career_outcomes
    TARGET_REGISTRY["annual_financials"] = get_annual_financials
    TARGET_REGISTRY["peak_overall"] = get_peak_overall
    TARGET_REGISTRY["merit_cap_share"] = get_merit_cap_share
    TARGET_REGISTRY["economic_targets"] = get_economic_targets
    TARGET_REGISTRY["career_value_metrics"] = get_career_value_metrics
    TARGET_REGISTRY["awards"] = get_awards
    _BUILTINS_REGISTERED = True


def get_target(name: str, loader: FOF8Loader) -> pl.DataFrame:
    _register_builtin_targets()
    try:
        target_fn = TARGET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(TARGET_REGISTRY)) or "none"
        raise ValueError(f"Unknown target '{name}'. Available targets: {available}.") from exc

    return target_fn(loader)
