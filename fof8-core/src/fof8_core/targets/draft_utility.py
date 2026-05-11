"""Draft-utility target builders derived from rookie control-window ratings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from fof8_core.loader import FOF8Loader

DEFAULT_CONTROL_YEARS: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_CONTROL_DISCOUNTS: dict[int, float] = {
    1: 1.00,
    2: 0.95,
    3: 0.85,
    4: 0.75,
}

DRAFT_UTILITY_TARGET_COLUMNS = [
    "Control_Y1_Current_Overall",
    "Control_Y2_Current_Overall",
    "Control_Y3_Current_Overall",
    "Control_Y4_Current_Overall",
    "Control_Window_Mean_Current_Overall",
    "Control_Window_Discounted_Mean_Current_Overall",
]

DRAFT_UTILITY_OUTPUT_COLUMNS = [
    "Player_ID",
    *DRAFT_UTILITY_TARGET_COLUMNS,
]


def _validate_control_years(control_years: Sequence[int]) -> list[int]:
    years = [int(year) for year in control_years]
    if not years:
        raise ValueError("control_years must contain at least one year.")
    if any(year < 1 for year in years):
        raise ValueError("control_years must contain only positive integers.")
    if len(set(years)) != len(years):
        raise ValueError("control_years must not contain duplicate years.")
    if years != sorted(years):
        raise ValueError("control_years must be sorted in ascending order.")
    return years


def _validate_control_discounts(
    control_years: Sequence[int],
    control_discounts: Mapping[int, float] | None,
) -> dict[int, float]:
    if control_discounts is None:
        control_discounts = DEFAULT_CONTROL_DISCOUNTS

    discounts = {int(year): float(value) for year, value in control_discounts.items()}
    missing_years = [year for year in control_years if year not in discounts]
    if missing_years:
        raise ValueError(
            "control_discounts is missing values for control years "
            f"{missing_years}."
        )
    return discounts


def get_draft_utility_targets(
    loader: FOF8Loader,
    *,
    control_years: Sequence[int] = DEFAULT_CONTROL_YEARS,
    control_discounts: Mapping[int, float] | None = None,
) -> pl.DataFrame:
    """Return rookie control-window rating targets from exhibition Current_Overall.

    The per-year control columns preserve missing seasons as null. The mean and
    discounted-mean summary targets are only populated when every requested
    control year is observed, which keeps right-censored draft classes explicit.
    """

    years = _validate_control_years(control_years)
    discounts = _validate_control_discounts(years, control_discounts)
    year_col_map = {year: f"Control_Y{year}_Current_Overall" for year in years}

    rookies = (
        loader.scan_file("rookies.csv")
        .select("Player_ID", pl.col("Year").alias("Draft_Year"))
        .unique(subset=["Player_ID"], keep="first")
    )
    ratings = (
        loader.scan_file("player_ratings_season_*.csv")
        .filter(pl.col("Scouting") == "Exhibition")
        .select("Player_ID", "Year", "Current_Overall")
    )

    grouped = (
        rookies.join(ratings, on="Player_ID", how="left")
        .with_columns((pl.col("Year") - pl.col("Draft_Year") + 1).alias("Career_Year"))
        .filter(pl.col("Career_Year").is_null() | pl.col("Career_Year").is_in(years))
        .group_by("Player_ID")
        .agg(
            *[
                pl.col("Current_Overall")
                .filter(pl.col("Career_Year") == year)
                .max()
                .alias(output_col)
                for year, output_col in year_col_map.items()
            ]
        )
        .collect()
        .sort("Player_ID")
    )

    required_exprs = [pl.col(output_col).is_not_null() for output_col in year_col_map.values()]
    complete_expr = pl.all_horizontal(required_exprs)
    discount_weight_total = sum(discounts[year] for year in years)

    mean_numerator = sum(pl.col(year_col_map[year]) for year in years)
    discounted_numerator = sum(pl.col(year_col_map[year]) * discounts[year] for year in years)

    return grouped.with_columns(
        pl.when(complete_expr)
        .then(mean_numerator / len(years))
        .otherwise(pl.lit(None))
        .alias("Control_Window_Mean_Current_Overall"),
        pl.when(complete_expr)
        .then(discounted_numerator / discount_weight_total)
        .otherwise(pl.lit(None))
        .alias("Control_Window_Discounted_Mean_Current_Overall"),
    ).select(DRAFT_UTILITY_OUTPUT_COLUMNS)
