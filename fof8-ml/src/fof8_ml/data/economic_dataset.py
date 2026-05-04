"""Economic dataset builder for universal feature/target/metadata construction."""

import logging

import polars as pl
from fof8_core.features.draft_class import get_draft_class
from fof8_core.loader import FOF8Loader
from fof8_core.targets.career import get_career_outcomes
from fof8_core.targets.financial import get_merit_cap_share, get_peak_overall

from fof8_ml.data.categorical import bucket_rare_colleges, cast_categoricals_to_enum


def build_economic_dataset(
    raw_path: str,
    league_name: str,
    year_range: list[int],
    final_sim_year: int,
    positions: list[str] | None = None,
    active_team_id: int | None = None,
    merit_threshold: float = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build feature, target, and metadata frames for the economic model."""
    _ = final_sim_year
    loader = FOF8Loader(base_path=raw_path, league_name=league_name)

    if active_team_id is None:
        active_team_id = loader.get_active_team_id()
        if active_team_id is not None:
            logging.info(f"Discovered active team ID: {active_team_id}")

    df_peak = get_peak_overall(loader)
    df_merit = get_merit_cap_share(loader)
    df_outcomes = get_career_outcomes(loader)

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
            (pl.col("Peak_Overall") * pl.col("Career_Merit_Cap_Share")).alias("DPO"),
            (pl.col("Career_Merit_Cap_Share") > merit_threshold)
            .alias("Cleared_Sieve")
            .cast(pl.Int8),
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
        pl.col("Cleared_Sieve").fill_null(0).cast(pl.Int8),
        pl.col("Economic_Success").fill_null(0).cast(pl.Int8),
        pl.col("DPO").fill_null(0.0),
        pl.col("Positive_DPO").fill_null(0.0),
        pl.col("Career_Merit_Cap_Share").fill_null(0.0),
        pl.col("Positive_Career_Merit_Cap_Share").fill_null(0.0),
    )

    df_model = df_master.drop(["Player_ID", "Year", "First_Name", "Last_Name"])
    df_model = bucket_rare_colleges(df_model, min_count=(end_year - start_year + 1))
    df_model = cast_categoricals_to_enum(df_model)

    target_columns = [
        "Cleared_Sieve",
        "Economic_Success",
        "DPO",
        "Positive_DPO",
        "Career_Merit_Cap_Share",
        "Positive_Career_Merit_Cap_Share",
    ]
    X = df_model.drop(target_columns)
    y = df_model.select(target_columns)
    metadata = df_master.select(["Player_ID", "Year", "First_Name", "Last_Name"])
    return X, y, metadata
