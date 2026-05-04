"""Economic dataset builder for universal feature/target/metadata construction."""

import logging

import polars as pl
from fof8_core.features.draft_class import get_draft_class
from fof8_core.loader import FOF8Loader
from fof8_core.targets.draft_outcomes import (
    DRAFT_OUTCOME_TARGET_COLUMNS,
    get_draft_outcome_targets,
)

from fof8_ml.data.categorical import bucket_rare_colleges, cast_categoricals_to_enum


def build_economic_dataset(
    raw_path: str,
    league_name: str,
    year_range: list[int],
    positions: list[str] | None = None,
    active_team_id: int | None = None,
    merit_threshold: float = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build feature, target, and metadata frames for the economic model."""
    loader = FOF8Loader(base_path=raw_path, league_name=league_name)

    if active_team_id is None:
        active_team_id = loader.get_active_team_id()
        if active_team_id is not None:
            logging.info(f"Discovered active team ID: {active_team_id}")

    df_targets = get_draft_outcome_targets(loader, merit_threshold=merit_threshold)

    feature_dfs = []
    start_year, end_year = year_range
    for year in range(start_year, end_year + 1):
        try:
            feature_dfs.append(get_draft_class(loader, year, active_team_id=active_team_id))
        except Exception as e:
            logging.info(f"Skipping year {year}: {e}")

    df_all_features = pl.concat(feature_dfs)

    if positions:
        df_all_features = df_all_features.filter(pl.col("Position_Group").is_in(positions))

    df_master = df_all_features.join(df_targets, on="Player_ID", how="left")

    duplicate_cols = [col for col in df_master.columns if col.endswith("_right")]
    if duplicate_cols:
        raise ValueError(
            f"Detected duplicate columns with '_right' suffix after join: {duplicate_cols}. "
            "Check for overlapping metadata in features and targets."
        )

    df_master = df_master.with_columns(
        [pl.col(col).fill_null(0) for col in DRAFT_OUTCOME_TARGET_COLUMNS]
    ).with_columns(
        pl.col("Cleared_Sieve").cast(pl.Int8),
        pl.col("Economic_Success").cast(pl.Int8),
    )

    df_model = df_master.drop(["Player_ID", "Year", "First_Name", "Last_Name"])
    df_model = bucket_rare_colleges(df_model, min_count=(end_year - start_year + 1))
    df_model = cast_categoricals_to_enum(df_model)

    target_columns = DRAFT_OUTCOME_TARGET_COLUMNS
    X = df_model.drop(target_columns)
    y = df_model.select(target_columns)
    metadata = df_master.select(["Player_ID", "Year", "First_Name", "Last_Name"])
    return X, y, metadata
