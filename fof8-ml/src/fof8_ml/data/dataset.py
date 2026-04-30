import logging

import polars as pl
from fof8_core.features import get_draft_class
from fof8_core.loader import FOF8Loader
from fof8_core.targets import get_career_outcomes, get_merit_cap_share, get_peak_overall


# TODO: Determine the fate of `build_survival_dataset`.
# This function applies filtering and splitting (training domain logic) which might belong in
# `pipelines/process_features.py` or the orchestrator, rather than the core data module.
# Consider deprecating this or moving the temporal splitting logic upstream.
def build_survival_dataset(
    raw_path: str,
    league_name: str,
    year_range: list[int],
    final_sim_year: int,
    survival_threshold: int,
    target_column: str = "Career_Games_Played",
    positions: list[str] | None = None,
    active_team_id: int = None,
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Builds the feature matrix (X) and target vector (y) for the survival classifier.
    """
    loader = FOF8Loader(base_path=raw_path, league_name=league_name)

    # 0. Discovery: Find active team ID if not provided
    if active_team_id is None:
        active_team_id = loader.get_active_team_id()
        if active_team_id is not None:
            logging.info(f"Discovered active team ID: {active_team_id}")

    # 1. Fetch Career Targets
    df_targets = get_career_outcomes(loader)

    df_targets = df_targets.with_columns(
        (pl.col(target_column) >= survival_threshold).alias("Survived").cast(pl.Int8)
    ).select(["Player_ID", "Survived"])

    # 2. Fetch Features for the requested years
    feature_dfs = []
    start_year, end_year = year_range
    for year in range(start_year, end_year + 1):
        try:
            df_features = get_draft_class(loader, year, active_team_id=active_team_id)
            feature_dfs.append(df_features)
        except Exception as e:
            logging.info(f"Skipping year {year}: {e}")

    df_all_features = pl.concat(feature_dfs)

    if positions and positions != "all":
        if isinstance(positions, str):
            positions = [positions]
        df_all_features = df_all_features.filter(pl.col("Position_Group").is_in(positions))

    # 3. Join Features and Targets
    df_master = df_all_features.join(
        df_targets,
        on="Player_ID",
        how="left",  # Preserve all rookies
    )

    # 3.5 Handle the missing outcomes for undrafted busts
    df_master = df_master.with_columns(pl.col("Survived").fill_null(0).cast(pl.Int8))

    # 4. Prep for XGBoost
    # Drop identifiers to prevent data leakage
    df_model = df_master.drop(["Player_ID", "Year", "First_Name", "Last_Name"])

    # 5. Feature Engineering: Bucket rare colleges to prevent overfitting
    # and handle unseen categories
    if "College" in df_model.columns:
        college_counts = df_model["College"].value_counts()
        # Keep colleges with at least 10 prospects across the dataset
        top_colleges = college_counts.filter(pl.col("count") >= 10)["College"].to_list()
        df_model = df_model.with_columns(
            pl.when(pl.col("College").is_in(top_colleges))
            .then(pl.col("College"))
            .otherwise(pl.lit("Other"))
            .alias("College")
        )

    # Cast all string/categorical columns to Enums to ensure consistent domains across CV folds
    cat_cols = [
        col for col, dtype in df_model.schema.items() if dtype in [pl.String, pl.Categorical]
    ]

    for col in cat_cols:
        # Sort for determinism
        unique_categories = df_model.get_column(col).unique().sort().cast(pl.String)
        df_model = df_model.with_columns(pl.col(col).cast(pl.Enum(unique_categories)))

    # Separate features and target
    X = df_model.drop("Survived")
    y = df_model.get_column("Survived")

    return X, y


def build_economic_dataset(
    raw_path: str,
    league_name: str,
    year_range: list[int],
    final_sim_year: int,
    positions: list[str] | None = None,
    active_team_id: int = None,
    merit_threshold: float = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Builds the feature matrix (X) and target DataFrame (y) for the two-stage economic model.
    """
    loader = FOF8Loader(base_path=raw_path, league_name=league_name)

    # 0. Discovery: Find active team ID if not provided
    if active_team_id is None:
        active_team_id = loader.get_active_team_id()
        if active_team_id is not None:
            logging.info(f"Discovered active team ID: {active_team_id}")

    # 1. Fetch Targets
    df_peak = get_peak_overall(loader)
    df_merit = get_merit_cap_share(loader)
    df_outcomes = get_career_outcomes(loader)

    # 2. Join Target components into a single truth table.
    # We use a union of all Player_IDs to ensure we don't lose anyone during the join,
    # then left join the specific target dataframes.
    all_ids = pl.concat(
        [df_peak.select("Player_ID"), df_merit.select("Player_ID"), df_outcomes.select("Player_ID")]
    ).unique()

    df_targets = (
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
            # Stage 2 Target: Intensity is Peak multiplied by Pure Financial Merit
            (pl.col("Peak_Overall") * pl.col("Career_Merit_Cap_Share")).alias("DGO"),
            # Stage 1 Target: Did they earn more than they were handed? (> 0)
            (pl.col("Career_Merit_Cap_Share") > merit_threshold)
            .alias("Cleared_Sieve")
            .cast(pl.Int8),
        )
        .select(
            [
                "Player_ID",
                "Cleared_Sieve",
                "DGO",
                "Career_Merit_Cap_Share",
                "Peak_Overall",
                "Career_Games_Played",
            ]
        )
    )

    # 3. Fetch Features for the requested years
    feature_dfs = []
    start_year, end_year = year_range
    for year in range(start_year, end_year + 1):
        try:
            df_features = get_draft_class(loader, year, active_team_id=active_team_id)
            feature_dfs.append(df_features)
        except Exception as e:
            logging.info(f"Skipping year {year}: {e}")

    df_all_features = pl.concat(feature_dfs)

    if positions and positions != "all":
        if isinstance(positions, str):
            positions = [positions]
        df_all_features = df_all_features.filter(pl.col("Position_Group").is_in(positions))

    # 4. Join Features and Targets
    df_master = df_all_features.join(
        df_targets,
        on="Player_ID",
        how="left",  # Preserve all rookies
    )

    # 4.5 Check for duplicate columns (join collisions)
    duplicate_cols = [col for col in df_master.columns if col.endswith("_right")]
    if duplicate_cols:
        raise ValueError(
            f"Detected duplicate columns with '_right' suffix after join: {duplicate_cols}. "
            "Check for overlapping metadata in features and targets."
        )

    # 3.5 Handle the missing outcomes for undrafted busts
    df_master = df_master.with_columns(
        # If they didn't make the target table, they didn't clear the sieve
        pl.col("Cleared_Sieve").fill_null(0).cast(pl.Int8),
        # Undrafted players generated 0 financial merit
        pl.col("DGO").fill_null(0.0),
        pl.col("Career_Merit_Cap_Share").fill_null(0.0),
    )

    # 4. Prep for XGBoost/CatBoost
    df_model = df_master.drop(["Player_ID", "Year", "First_Name", "Last_Name"])

    # 5. Feature Engineering: Bucket rare colleges to prevent overfitting
    # and handle unseen categories
    if "College" in df_model.columns:
        college_counts = df_model["College"].value_counts()
        # Keep colleges with at least 1 prospects per year on average
        top_colleges = college_counts.filter(pl.col("count") >= 1 * (end_year - start_year + 1))[
            "College"
        ].to_list()
        df_model = df_model.with_columns(
            pl.when(pl.col("College").is_in(top_colleges))
            .then(pl.col("College"))
            .otherwise(pl.lit("Other"))
            .alias("College")
        )

    cat_cols = [
        col for col, dtype in df_model.schema.items() if dtype in [pl.String, pl.Categorical]
    ]

    for col in cat_cols:
        unique_categories = df_model.get_column(col).unique().sort().cast(pl.String)
        df_model = df_model.with_columns(pl.col(col).cast(pl.Enum(unique_categories)))

    X = df_model.drop(["Cleared_Sieve", "DGO", "Career_Merit_Cap_Share"])
    y = df_model.select(["Cleared_Sieve", "DGO", "Career_Merit_Cap_Share"])
    metadata = df_master.select(["Player_ID", "Year", "First_Name", "Last_Name"])

    return X, y, metadata
