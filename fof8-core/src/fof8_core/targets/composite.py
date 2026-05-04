"""Composite target builders that intentionally combine target families."""

import polars as pl

from fof8_core.loader import FOF8Loader
from fof8_core.targets.financial import get_merit_cap_share
from fof8_core.targets.talent import get_peak_overall

COMPOSITE_TARGET_COLUMNS = [
    "DPO",
    "Positive_DPO",
]

COMPOSITE_SOURCE_COLUMNS = [
    "Peak_Overall",
    "Career_Merit_Cap_Share",
]

COMPOSITE_OUTPUT_COLUMNS = [
    "Player_ID",
    *COMPOSITE_TARGET_COLUMNS,
]


def add_dpo_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Derive DPO columns from talent and economic source columns."""
    return df.with_columns(
        [
            pl.col("Peak_Overall").fill_null(0),
            pl.col("Career_Merit_Cap_Share").fill_null(0),
        ]
    ).with_columns(
        (pl.col("Peak_Overall") * pl.col("Career_Merit_Cap_Share")).alias("DPO"),
        (pl.col("Peak_Overall") * pl.col("Career_Merit_Cap_Share"))
        .clip(lower_bound=0.0)
        .alias("Positive_DPO"),
    )


def get_dpo_targets(loader: FOF8Loader) -> pl.DataFrame:
    """Build DPO composite targets by joining talent and merit sources."""
    df_peak = get_peak_overall(loader, k=3)
    df_merit = get_merit_cap_share(loader)

    all_ids = pl.concat(
        [
            df_peak.select("Player_ID"),
            df_merit.select("Player_ID"),
        ]
    ).unique()

    base_df = all_ids.join(
        df_peak.select(["Player_ID", "Peak_Overall"]), on="Player_ID", how="left"
    ).join(df_merit.select(["Player_ID", "Career_Merit_Cap_Share"]), on="Player_ID", how="left")

    return add_dpo_columns(base_df).select(COMPOSITE_OUTPUT_COLUMNS)
