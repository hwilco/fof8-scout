"""Broad draft-outcome target bundle for ML dataset generation."""

import polars as pl

from fof8_core.loader import FOF8Loader
from fof8_core.targets.career import get_career_outcomes
from fof8_core.targets.composite import COMPOSITE_TARGET_COLUMNS, get_dpo_targets
from fof8_core.targets.economic import ECONOMIC_TARGET_COLUMNS, get_economic_targets
from fof8_core.targets.talent import TALENT_TARGET_COLUMNS, get_peak_overall

DRAFT_OUTCOME_TARGET_COLUMNS = [
    *ECONOMIC_TARGET_COLUMNS,
    *COMPOSITE_TARGET_COLUMNS,
]

DRAFT_OUTCOME_CONTEXT_COLUMNS = [
    *TALENT_TARGET_COLUMNS,
    "Career_Games_Played",
]

# Canonical processed outcome columns. These are always excluded from training
# features and also form the baseline cross-outcome scorecard contract.
DRAFT_OUTCOME_LEAKAGE_COLUMNS = [
    *DRAFT_OUTCOME_TARGET_COLUMNS,
    *DRAFT_OUTCOME_CONTEXT_COLUMNS,
]

DRAFT_OUTCOME_OUTPUT_COLUMNS = [
    "Player_ID",
    *DRAFT_OUTCOME_TARGET_COLUMNS,
    *DRAFT_OUTCOME_CONTEXT_COLUMNS,
]


def get_draft_outcome_targets(
    loader: FOF8Loader,
    merit_threshold: float = 0,
) -> pl.DataFrame:
    """Build the ML-ready draft outcome frame from target-family components."""
    df_economic = get_economic_targets(loader, merit_threshold=merit_threshold)
    df_dpo = get_dpo_targets(loader)
    df_peak = get_peak_overall(loader, k=3)
    df_outcomes = get_career_outcomes(loader).select(["Player_ID", "Career_Games_Played"])

    all_ids = pl.concat(
        [
            df_economic.select("Player_ID"),
            df_dpo.select("Player_ID"),
            df_peak.select("Player_ID"),
            df_outcomes.select("Player_ID"),
        ]
    ).unique()

    out = (
        all_ids.join(df_economic, on="Player_ID", how="left")
        .join(df_dpo, on="Player_ID", how="left")
        .join(df_peak.select(["Player_ID", "Peak_Overall"]), on="Player_ID", how="left")
        .join(df_outcomes, on="Player_ID", how="left")
    )

    return out.with_columns(
        [pl.col(col).fill_null(0) for col in DRAFT_OUTCOME_LEAKAGE_COLUMNS]
    ).select(DRAFT_OUTCOME_OUTPUT_COLUMNS)
