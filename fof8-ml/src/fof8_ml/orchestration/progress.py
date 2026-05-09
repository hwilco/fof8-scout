from __future__ import annotations

import polars as pl


def _count_universes(meta: pl.DataFrame) -> int:
    if len(meta) == 0 or "Universe" not in meta.columns:
        return 0
    return int(meta.get_column("Universe").n_unique())


def print_validation_run_start(
    *,
    role_label: str,
    meta_train: pl.DataFrame,
    meta_val: pl.DataFrame,
    meta_test: pl.DataFrame,
    extra_detail: str | None = None,
) -> None:
    """Print the starting summary for a held-out validation run."""
    print(
        f"\nRunning {role_label} with held-out universes: "
        f"train={_count_universes(meta_train)} ({len(meta_train)} rows), "
        f"val={_count_universes(meta_val)} ({len(meta_val)} rows), "
        f"test={_count_universes(meta_test)} ({len(meta_test)} rows)."
    )
    if extra_detail:
        print(extra_detail)


def print_phase(message: str) -> None:
    """Print a concise pipeline phase update."""
    print(f"\n[{message}]")


def print_refit_phase(role_label: str, train_rows: int, val_rows: int) -> None:
    """Print the refit phase for a final train+validation model."""
    print(
        f"\n[Refitting final {role_label} on train+validation universes "
        f"({train_rows + val_rows} rows total)]"
    )


def print_test_phase(role_label: str, test_rows: int) -> None:
    """Print the held-out test scoring phase."""
    print(f"\n[Scoring held-out test universes for {role_label} ({test_rows} rows)]")
