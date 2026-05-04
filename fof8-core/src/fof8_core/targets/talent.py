"""Talent target builders for scouted/rating-based outcomes."""

import polars as pl

from fof8_core.loader import FOF8Loader

TALENT_TARGET_COLUMNS = [
    "Peak_Overall",
]

TALENT_OUTPUT_COLUMNS = [
    "Player_ID",
    *TALENT_TARGET_COLUMNS,
]


def get_peak_overall(loader: FOF8Loader, k: int = 1, timeframe: int | None = None) -> pl.DataFrame:
    """
    Calculates the mean of the top k Current_Overall values for each player,
    using the league-scout post-exhibition ratings.

    Args:
        loader: An instance of FOF8Loader to access the data.
        k: The number of top ratings to consider for calculating the peak overall. Default is 1
            (i.e., the single highest rating).
        timeframe: Optional number of career years (starting from each player's first
            observed exhibition rating year) to include. If None, considers all
            available seasons. Default is None.
    Returns:
        A DataFrame with Player_ID and their corresponding Peak_Overall rating.
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
            .agg(pl.col("Current_Overall").top_k(k).mean().alias("Peak_Overall"))
            .collect()
        )
