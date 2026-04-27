"""
Strategies for calculating replacement-level value for use in Value Over Replacement Player (VORP) metrics.
"""

from typing import Callable
import polars as pl

from .schemas import POSITION_GROUPS

# Define the type alias for our replacement strategy functions
ReplacementStrategy = Callable[[pl.LazyFrame], pl.LazyFrame]


def strategy_25th_percentile(lf_av: pl.LazyFrame) -> pl.LazyFrame:
    """Calculates replacement level as the 25th percentile of active producers (AV > 0), per position group."""
    baseline = (
        lf_av.filter(pl.col("Season_AV") > 0)
        .group_by(["Year", "Position_Group"])
        .agg(pl.col("Season_AV").quantile(0.25).alias("Replacement_AV"))
    )

    return (
        lf_av.join(baseline, on=["Year", "Position_Group"], how="left")
        .with_columns(pl.col("Replacement_AV").fill_null(0.0))
        .with_columns((pl.col("Season_AV") - pl.col("Replacement_AV")).alias("Season_VORP"))
    )


def strategy_n_plus_one(lf_av: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates replacement level using the N+1 starter threshold for 32 teams.

    This strategy is 'safe' for low player counts: if a position group has fewer
    than the theoretical number of starters (e.g., Fullbacks), it falls back
    to using the lowest-ranked producer as the replacement level.
    """
    NUM_TEAMS = 32
    # Define starters per team for each position group
    # Using 2 OLB and 1 ILB to total the 3 Linebackers per team
    starters_per_position_group_per_team = [
        ("QB", 1),
        ("RB", 1),
        ("FB", 1),
        ("WR", 3),
        ("TE", 1),
        ("T", 2),
        ("G", 2),
        ("C", 1),
        ("DE", 2),
        ("DT", 2),
        ("OLB", 2),
        ("ILB", 1),
        ("CB", 2),
        ("S", 2),
        ("K", 1),
        ("P", 1),
        ("LS", 1),
    ]
    starters_per_position_group = [
        (position, NUM_TEAMS * starters_per_team)
        for position, starters_per_team in starters_per_position_group_per_team
    ]
    starter_counts = pl.DataFrame(
        starters_per_position_group,
        schema={"Position_Group": POSITION_GROUPS, "League_Starters": pl.Int32},
    ).lazy()

    baseline = (
        lf_av.filter(pl.col("Season_AV") > 0)
        .with_columns(
            pl.col("Season_AV")
            .rank(method="ordinal", descending=True)
            .over(["Year", "Position_Group"])
            .alias("Pos_Rank")
        )
        .with_columns(
            pl.col("Pos_Rank").max().over(["Year", "Position_Group"]).alias("Max_Pos_Rank")
        )
        .join(starter_counts, on="Position_Group", how="left")
        .with_columns(
            # Target the N+1th player, but cap at the total number of players available
            pl.min_horizontal(pl.col("League_Starters") + 1, pl.col("Max_Pos_Rank")).alias(
                "Target_Rank"
            )
        )
        .filter(pl.col("Pos_Rank") == pl.col("Target_Rank"))
        .select(["Year", "Position_Group", pl.col("Season_AV").alias("Replacement_AV")])
    )

    return (
        lf_av.join(baseline, on=["Year", "Position_Group"], how="left")
        .with_columns(pl.col("Replacement_AV").fill_null(0.0))
        .with_columns((pl.col("Season_AV") - pl.col("Replacement_AV")).alias("Season_VORP"))
    )


def strategy_hybrid_baseline(lf_av: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates replacement level using a rate-based (per-snap) bifurcated approach
    to solve the 'bench penalty' in zero-injury simulations.

    - Ironman Positions (QB, OL, Specialists): Uses the median AV/Snap of the 5 worst true starters.
    - Rotational Positions (RB, WR, DL, etc.): Uses the median AV/Snap of the 5th-25th percentile of snaps.
    """

    ironman_starters = {"QB": 1, "C": 1, "G": 2, "T": 2, "K": 1, "P": 1, "LS": 1}
    NUM_TEAMS = 32

    ironman_counts = pl.DataFrame(
        [
            {"Position_Group": pos, "League_Starters": count * NUM_TEAMS}
            for pos, count in ironman_starters.items()
        ],
        schema={"Position_Group": POSITION_GROUPS, "League_Starters": pl.Int32},
    ).lazy()

    # 1. Prepare player-level snap and rate data
    lf = lf_av.with_columns(
        pl.when(pl.col("Position_Group").is_in(["K", "P", "LS"]))
        .then(pl.col("S_Special_Teams_Plays"))
        .otherwise(pl.col("S_Pass_Plays") + pl.col("S_Run_Plays"))
        .alias("Relevant_Snaps")
    ).with_columns(
        # FIX: Require 10 snaps to form a stable rate, and floor the rate at 0.0
        # Net-negative players are below replacement, not the definition of replacement.
        pl.when(pl.col("Relevant_Snaps") >= 10)
        .then(pl.max_horizontal(pl.col("Season_AV") / pl.col("Relevant_Snaps"), 0.0))
        .otherwise(0.0)
        .alias("AV_Per_Snap")
    )

    # 2. Proxy Baseline (For Ironmen in Zero-Injury Sims)
    ironman_baseline = (
        lf.filter(pl.col("Position_Group").is_in(list(ironman_starters.keys())))
        .filter(pl.col("Relevant_Snaps") > 0)
        .with_columns(
            # FIX: Re-added the Max_Pos_Snaps filter to prevent 1-snap jumbo package anomalies
            pl.col("Relevant_Snaps").max().over(["Year", "Position_Group"]).alias("Max_Pos_Snaps")
        )
        .filter(pl.col("Relevant_Snaps") > (pl.col("Max_Pos_Snaps") * 0.25))
        .with_columns(
            pl.col("Season_AV")
            .rank(method="ordinal", descending=True)
            .over(["Year", "Position_Group"])
            .alias("Valid_Starter_Rank")
        )
        .with_columns(
            pl.col("Valid_Starter_Rank")
            .max()
            .over(["Year", "Position_Group"])
            .alias("Valid_Starter_Count")
        )
        .filter(pl.col("Valid_Starter_Rank") > (pl.col("Valid_Starter_Count") - 5))
        .group_by(["Year", "Position_Group"])
        .agg(pl.col("AV_Per_Snap").median().alias("Ironman_Rep_Rate"))
    )

    # 3. Snap Baseline (For Rotational Players)
    snap_baseline = (
        lf.filter(~pl.col("Position_Group").is_in(list(ironman_starters.keys())))
        .filter(pl.col("Relevant_Snaps") > 0)
        .with_columns(
            pl.col("Relevant_Snaps")
            .rank(method="min", descending=False)
            .over(["Year", "Position_Group"])
            .alias("Snap_Rank"),
            pl.len().over(["Year", "Position_Group"]).alias("Total_Active"),
        )
        .with_columns((pl.col("Snap_Rank") / pl.col("Total_Active")).alias("Snap_Percentile"))
        .filter((pl.col("Snap_Percentile") > 0.05) & (pl.col("Snap_Percentile") <= 0.25))
        .group_by(["Year", "Position_Group"])
        .agg(pl.col("AV_Per_Snap").median().alias("Snap_Rep_Rate"))
    )

    # 4. Merge, Route, and Calculate Player VORP
    return (
        lf.join(ironman_baseline, on=["Year", "Position_Group"], how="left")
        .join(snap_baseline, on=["Year", "Position_Group"], how="left")
        .with_columns(
            pl.when(pl.col("Position_Group").is_in(list(ironman_starters.keys())))
            .then(pl.col("Ironman_Rep_Rate"))
            .otherwise(pl.col("Snap_Rep_Rate"))
            .alias("Replacement_Rate")
        )
        .with_columns(pl.col("Replacement_Rate").fill_null(0.0))
        .with_columns(
            (pl.col("Relevant_Snaps") * pl.col("Replacement_Rate")).alias("Replacement_AV")
        )
        .with_columns((pl.col("Season_AV") - pl.col("Replacement_AV")).alias("Season_VORP"))
        .drop(
            [
                "Relevant_Snaps",
                "AV_Per_Snap",
                "Ironman_Rep_Rate",
                "Snap_Rep_Rate",
                "Replacement_Rate",
            ]
        )
    )
