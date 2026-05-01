"""Builds a binary career-threshold training dataset from rookie features and outcomes.

The builder labels each prospect as clearing or not clearing a configurable
career threshold (for example, minimum games played or seasons). It:
1. Loads rookie draft-class features over a year range.
2. Loads career outcomes and converts a selected outcome column into a binary label.
3. Left-joins outcomes onto rookies so undrafted/missing outcomes remain in the dataset.
4. Fills missing labels to 0, applies categorical bucketing/enum casting, and returns
   (X, y) for classifier training.
"""

import logging

import polars as pl
from fof8_core.features.draft_class import get_draft_class
from fof8_core.loader import FOF8Loader
from fof8_core.targets.career import get_career_outcomes

from fof8_ml.data.categorical import bucket_rare_colleges, cast_categoricals_to_enum


def build_career_threshold_dataset(
    raw_path: str,
    league_name: str,
    year_range: list[int],
    final_sim_year: int,
    career_threshold: int,
    target_column: str = "Career_Games_Played",
    positions: list[str] | None = None,
    active_team_id: int = None,
) -> tuple[pl.DataFrame, pl.Series]:
    """Return classifier inputs for binary prospect career-threshold prediction.

    A row is labeled `Cleared_Career_Threshold=1` when
    `target_column >= career_threshold`, otherwise `0`.
    """
    _ = final_sim_year
    loader = FOF8Loader(base_path=raw_path, league_name=league_name)

    if active_team_id is None:
        active_team_id = loader.get_active_team_id()
        if active_team_id is not None:
            logging.info(f"Discovered active team ID: {active_team_id}")

    df_targets = get_career_outcomes(loader)
    df_targets = df_targets.with_columns(
        (pl.col(target_column) >= career_threshold).alias("Cleared_Career_Threshold").cast(pl.Int8)
    ).select(["Player_ID", "Cleared_Career_Threshold"])

    feature_dfs = []
    start_year, end_year = year_range
    for year in range(start_year, end_year + 1):
        try:
            feature_dfs.append(get_draft_class(loader, year, active_team_id=active_team_id))
        except Exception as e:
            logging.info(f"Skipping year {year}: {e}")

    df_all_features = pl.concat(feature_dfs)

    if positions and positions != "all":
        if isinstance(positions, str):
            positions = [positions]
        df_all_features = df_all_features.filter(pl.col("Position_Group").is_in(positions))

    df_master = df_all_features.join(df_targets, on="Player_ID", how="left")
    df_master = df_master.with_columns(
        pl.col("Cleared_Career_Threshold").fill_null(0).cast(pl.Int8)
    )

    df_model = df_master.drop(["Player_ID", "Year", "First_Name", "Last_Name"])
    df_model = bucket_rare_colleges(df_model, min_count=10)
    df_model = cast_categoricals_to_enum(df_model)

    X = df_model.drop("Cleared_Career_Threshold")
    y = df_model.get_column("Cleared_Career_Threshold")
    return X, y
