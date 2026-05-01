"""Approximate value and replacement-adjusted career value target builders."""

import polars as pl

from fof8_core.loader import FOF8Loader
from fof8_core.targets_replacements import ReplacementStrategy, strategy_hybrid_baseline


def calculate_season_av(lf_records: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates a universal Approximate Value (AV) production score for each player-season
    based on raw box-score statistics.
    """
    return (
        lf_records.with_columns(
            [
                # Fill nulls for all season stats to prevent math errors
                pl.col("^S_.*$").fill_null(0).cast(pl.Float32)
            ]
        )
        .with_columns(
            (
                # Passing
                (pl.col("S_Passing_Yards") / 25)
                + (pl.col("S_Touchdown_Passes") * 4)
                - (pl.col("S_Intercepted") * 2)
                - (pl.col("S_Sacked") * 1)
                +
                # Rushing & Receiving
                (pl.col("S_Rushing_Yards") / 10)
                + (pl.col("S_Receiving_Yards") / 10)
                + (pl.col("S_Rushing_Touchdowns") * 6)
                + (pl.col("S_Receiving_Touchdowns") * 6)
                +
                # Defense
                (pl.col("S_Tackles") * 1)
                + (pl.col("S_Assists") * 0.5)
                + ((pl.col("S_Sacks_(x10)") / 10) * 4)
                + (pl.col("S_Interceptions") * 5)
                + (pl.col("S_Passes_Defensed") * 1)
                +
                # Turnovers
                (pl.col("S_Fumbles_Forced") * 3)
                + (pl.col("S_Fumbles_Recovered") * 3)
                - (pl.col("S_Fumbles") * 2)
                +
                # Blocking (Applies to TEs, FBs, WRs, and OL)
                (pl.col("S_Key_Run_Blocks") * 0.5)
                + (pl.col("S_Pancake_Blocks") * 1)
                - (pl.col("S_Sacks_Allowed") * 3)
                +
                # Special Teams / Specialists
                (pl.col("S_Field_Goals_Made") * 3)
                - ((pl.col("S_Field_Goals_Attempted") - pl.col("S_Field_Goals_Made")) * 2)
                + (pl.col("S_Punts_Inside_20") * 1)
                + (pl.col("S_Punt_Returns_Touchdowns") * 6)
                + (pl.col("S_Kick_Return_Touchdowns") * 6)
            ).alias("Base_Stat_AV")
        )
        .with_columns(
            # Offensive Line Proxy: Add start multipliers ONLY for true O-Linemen
            # since they cannot accumulate yardage or tackle stats.
            pl.when(pl.col("Position_Group").is_in(["C", "G", "T"]))
            .then(pl.col("Base_Stat_AV") + (pl.col("S_Games_Started") * 2.5))
            .otherwise(pl.col("Base_Stat_AV"))
            .alias("Season_AV")
        )
    )


def get_career_value_metrics(
    loader: FOF8Loader, strategy: ReplacementStrategy = strategy_hybrid_baseline
) -> pl.DataFrame:
    """
    Calculates universal Approximate Value (AV) and position-adjusted
    Value Over Replacement Player (VORP) using the specified strategy.
    """
    with pl.StringCache():
        lf_records = loader.scan_file("player_record.csv")

        # 1. Calculate Season AV
        lf_av = calculate_season_av(lf_records)

        # 2. Calculate Season-by-Season VORP using the specified strategy
        # Strategies now return the full dataframe with Season_VORP pre-calculated.
        lf_vorp = strategy(lf_av)

        # 3. Aggregate to Career Totals
        return (
            lf_vorp.group_by("Player_ID")
            .agg(
                [
                    pl.col("Season_AV").sum().alias("Career_AV"),
                    pl.col("Season_VORP").sum().alias("Career_VORP"),
                ]
            )
            .collect()
        )
