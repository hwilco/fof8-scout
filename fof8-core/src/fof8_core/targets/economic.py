"""Economic target builders for downstream ML datasets."""

import polars as pl

from fof8_core.loader import FOF8Loader
from fof8_core.targets.financial import get_merit_cap_share

ECONOMIC_TARGET_COLUMNS = [
    "Cleared_Sieve",
    "Economic_Success",
    "Career_Merit_Cap_Share",
    "Positive_Career_Merit_Cap_Share",
]

ECONOMIC_SOURCE_COLUMNS = [
    "Career_Merit_Cap_Share",
]

ECONOMIC_OUTPUT_COLUMNS = [
    "Player_ID",
    *ECONOMIC_TARGET_COLUMNS,
]


def add_economic_derived_columns(df: pl.DataFrame, merit_threshold: float = 0) -> pl.DataFrame:
    """Derive canonical economic target columns from source economic context columns."""
    return df.with_columns(
        [
            pl.col("Career_Merit_Cap_Share").fill_null(0),
        ]
    ).with_columns(
        (pl.col("Career_Merit_Cap_Share") > merit_threshold).alias("Cleared_Sieve").cast(pl.Int8),
        pl.col("Career_Merit_Cap_Share")
        .clip(lower_bound=0.0)
        .alias("Positive_Career_Merit_Cap_Share"),
        (pl.col("Career_Merit_Cap_Share") > 0).alias("Economic_Success").cast(pl.Int8),
    )


def get_economic_targets(loader: FOF8Loader, merit_threshold: float = 0) -> pl.DataFrame:
    """Build canonical economic targets from core target sources."""
    df_merit = get_merit_cap_share(loader)

    base_df = df_merit.select(["Player_ID", "Career_Merit_Cap_Share"])
    return add_economic_derived_columns(base_df, merit_threshold=merit_threshold).select(
        ECONOMIC_OUTPUT_COLUMNS
    )
