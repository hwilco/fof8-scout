"""Economic target builders for downstream ML datasets."""

import polars as pl

from fof8_core.loader import FOF8Loader
from fof8_core.targets.career import get_career_outcomes
from fof8_core.targets.financial import get_merit_cap_share, get_peak_overall


def get_economic_targets(loader: FOF8Loader, merit_threshold: float = 0) -> pl.DataFrame:
    """Build canonical economic targets from core target sources."""
    df_peak = get_peak_overall(loader)
    df_merit = get_merit_cap_share(loader)
    df_outcomes = get_career_outcomes(loader)

    all_ids = pl.concat(
        [
            df_peak.select("Player_ID"),
            df_merit.select("Player_ID"),
            df_outcomes.select("Player_ID"),
        ]
    ).unique()

    return (
        all_ids.join(df_peak.select(["Player_ID", "Peak_Overall"]), on="Player_ID", how="left")
        .join(df_merit.select(["Player_ID", "Career_Merit_Cap_Share"]), on="Player_ID", how="left")
        .join(df_outcomes.select(["Player_ID", "Career_Games_Played"]), on="Player_ID", how="left")
        .with_columns(
            [
                pl.col("Peak_Overall").fill_null(0),
                pl.col("Career_Merit_Cap_Share").fill_null(0),
                pl.col("Career_Games_Played").fill_null(0),
            ]
        )
        .with_columns(
            (pl.col("Peak_Overall") * pl.col("Career_Merit_Cap_Share")).alias("DPO"),
            (pl.col("Career_Merit_Cap_Share") > merit_threshold).alias("Cleared_Sieve").cast(pl.Int8),
            pl.col("Career_Merit_Cap_Share")
            .clip(lower_bound=0.0)
            .alias("Positive_Career_Merit_Cap_Share"),
            (pl.col("Peak_Overall") * pl.col("Career_Merit_Cap_Share"))
            .clip(lower_bound=0.0)
            .alias("Positive_DPO"),
            (pl.col("Career_Merit_Cap_Share") > 0).alias("Economic_Success").cast(pl.Int8),
        )
        .select(
            [
                "Player_ID",
                "Cleared_Sieve",
                "Economic_Success",
                "DPO",
                "Positive_DPO",
                "Career_Merit_Cap_Share",
                "Positive_Career_Merit_Cap_Share",
                "Peak_Overall",
                "Career_Games_Played",
            ]
        )
    )
