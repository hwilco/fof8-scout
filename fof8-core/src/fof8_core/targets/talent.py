"""Talent target builders for scouted/rating-based outcomes."""

import polars as pl

from fof8_core.loader import FOF8Loader

TALENT_TARGET_COLUMNS = [
    "Peak_Overall",
    "Top3_Mean_Current_Overall",
]

TALENT_OUTPUT_COLUMNS = [
    "Player_ID",
    *TALENT_TARGET_COLUMNS,
]


def _get_topk_mean_current_overall(
    loader: FOF8Loader,
    *,
    k: int,
    output_col: str,
    timeframe: int | None = None,
) -> pl.DataFrame:
    """
    Calculates a top-k Current_Overall summary for each player using exhibition ratings.

    Args:
        loader: An instance of FOF8Loader to access the data.
        k: The number of top ratings to average.
        output_col: Output column name for the aggregated talent target.
        timeframe: Optional number of career years (starting from each player's first
            observed exhibition rating year) to include. If None, considers all
            available seasons. Default is None.
    Returns:
        A DataFrame with Player_ID and the requested talent target column.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if timeframe is not None and timeframe < 1:
        raise ValueError("timeframe must be >= 1 when provided.")

    with pl.StringCache():
        lf_ratings = loader.scan_file("player_ratings_season_*.csv")
        lf_exhibition = lf_ratings.filter(pl.col("Scouting") == "Exhibition").select(
            ["Player_ID", "Year", "Current_Overall"]
        )

        if timeframe is not None:
            lf_exhibition = (
                lf_exhibition.with_columns(
                    pl.col("Year").min().over("Player_ID").alias("First_Year")
                )
                .filter(pl.col("Year") <= (pl.col("First_Year") + timeframe - 1))
                .drop("First_Year")
            )

        return (
            lf_exhibition.group_by("Player_ID")
            .agg(pl.col("Current_Overall").top_k(k).mean().alias(output_col))
            .collect()
        )


def get_peak_overall(loader: FOF8Loader, timeframe: int | None = None) -> pl.DataFrame:
    """Return each player's single highest exhibition Current_Overall rating."""
    return _get_topk_mean_current_overall(
        loader,
        k=1,
        output_col="Peak_Overall",
        timeframe=timeframe,
    )


def get_top3_mean_current_overall(
    loader: FOF8Loader,
    timeframe: int | None = None,
) -> pl.DataFrame:
    """Return the mean of each player's top-3 exhibition Current_Overall ratings."""
    return _get_topk_mean_current_overall(
        loader,
        k=3,
        output_col="Top3_Mean_Current_Overall",
        timeframe=timeframe,
    )
