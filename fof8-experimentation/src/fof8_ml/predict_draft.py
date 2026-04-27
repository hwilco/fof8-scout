import hydra
import mlflow
import polars as pl
import numpy as np
import random
from pathlib import Path
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd

from fof8_core.loader import FOF8Loader
from fof8_core.features import get_draft_class


@hydra.main(version_base=None, config_path="../../conf", config_name="economic_pipeline")
def main(cfg: DictConfig):
    # Define a stable root directory for the experimentation package (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Set the tracking URI to be stable within the experimentation package
    db_path = os.path.join(exp_root, "mlflow.db")
    tracking_uri = f"sqlite:///{db_path.replace('\\', '/')}"
    print(f"Using tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    # Parameters (configurable via CLI overrides: run_id=..., league=..., n_years=...)
    parent_run_id = cfg.get("run_id", "fd6b52330ac34dc2b816cef61fa92dfb")
    league_name = cfg.get("league", "DRAFT004")
    n_years_to_predict = cfg.get("n_years", 10)

    print(f"Targeting Parent Run ID: {parent_run_id}")
    print(f"Targeting League: {league_name}")

    client = mlflow.tracking.MlflowClient()

    # Get all experiment IDs
    experiment_ids = [e.experiment_id for e in client.search_experiments()]

    # Find the Stage 1 nested run
    nested_runs = client.search_runs(
        experiment_ids=experiment_ids, filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )

    stage1_run = next(
        (r for r in nested_runs if r.data.tags.get("mlflow.runName") == "Stage1_Sieve_Classifier"),
        None,
    )

    if not stage1_run:
        print(f"Could not find Stage 1 nested run for parent {parent_run_id}.")
        # Fallback: maybe the run_id provided IS the stage1 run?
        run_info = client.get_run(parent_run_id)
        if run_info.data.tags.get("mlflow.runName") == "Stage1_Sieve_Classifier":
            stage1_run = run_info
        else:
            print("Failed to find Stage 1 model.")
            return

    print(f"Loading Stage 1 Model from Run: {stage1_run.info.run_id}")
    stage1_model_uri = f"runs:/{stage1_run.info.run_id}/stage1_model"

    # Load the model
    # We need to know if it's CatBoost or XGBoost. We can check the run params or tags.
    # Or just try both.
    try:
        model = mlflow.catboost.load_model(stage1_model_uri)
        is_catboost = True
        print("Loaded CatBoost model.")
    except:
        try:
            model = mlflow.xgboost.load_model(stage1_model_uri)
            is_catboost = False
            print("Loaded XGBoost model.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

    # Initialize Loader
    absolute_raw_path = to_absolute_path(cfg.data.raw_path)
    loader = FOF8Loader(base_path=absolute_raw_path, league_name=league_name)
    active_team_id = loader.get_active_team_id()

    # Get available years
    available_years = sorted(
        [int(p.name) for p in loader.league_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    )

    if not available_years:
        print(f"No years found in {loader.league_dir}")
        return

    print(f"Found {len(available_years)} years. Selecting {n_years_to_predict} random years.")
    selected_years = random.sample(available_years, min(n_years_to_predict, len(available_years)))
    selected_years.sort()
    print(f"Selected years: {selected_years}")

    all_results = []

    for year in selected_years:
        print(f"Processing Year: {year}...")
        try:
            # 1. Fetch Features
            df_features = get_draft_class(loader, year, active_team_id=active_team_id)

            # 2. Fetch Age (needed if model uses it)
            # Based on inference.py, it joins with player_information.csv
            # We use 2144 as reference for player_info if needed, but usually we use the current year's data
            # for birth years.
            try:
                df_info = (
                    loader.scan_file("player_information.csv", year=year)
                    .select(["Player_ID", "Draft_Year", "Year_Born"])
                    .collect()
                )
                df_features = (
                    df_features.join(df_info, on="Player_ID", how="left")
                    .with_columns((pl.col("Draft_Year") - pl.col("Year_Born")).alias("Age"))
                    .drop(["Draft_Year", "Year_Born"])
                )
            except Exception as e:
                print(f"  Warning: Could not calculate Age for {year}: {e}")
                # Add dummy Age if missing? Or hope it's not needed.
                if "Age" not in df_features.columns:
                    df_features = df_features.with_columns(pl.lit(22).alias("Age"))

            # 3. Preprocess for model
            # Drop non-feature columns
            cols_to_drop = ["Player_ID", "Year", "First_Name", "Last_Name"]
            df_model = df_features.drop([c for c in cols_to_drop if c in df_features.columns])

            X_pd = df_model.to_pandas()

            # Handle Categoricals
            if "College" in X_pd.columns:
                # Training script might have mapped colleges. inference.py sets to 'Other'
                X_pd["College"] = "Other"

            cat_cols = X_pd.select_dtypes(include=["object"]).columns.tolist()
            for col in cat_cols:
                X_pd[col] = X_pd[col].astype("category")

            # 4. Predict
            if is_catboost:
                probs = model.predict_proba(X_pd)[:, 1]
            else:
                probs = model.predict_proba(X_pd)[:, 1]

            # 5. Store Results
            df_year_results = df_features.select(
                [
                    pl.col("Player_ID"),
                    pl.col("Year"),
                    pl.col("First_Name"),
                    pl.col("Last_Name"),
                    pl.col("Position_Group"),
                ]
            ).with_columns([pl.Series("P_Sieve", probs)])

            all_results.append(df_year_results)

        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            continue

    if not all_results:
        print("No results generated.")
        return

    df_final = pl.concat(all_results)

    # Sort by probability within each year/position or overall?
    df_final = df_final.sort(["Year", "P_Sieve"], descending=[False, True])

    output_file = to_absolute_path(f"draft004_random10_predictions.csv")
    df_final.write_csv(output_file)

    print("\n" + "=" * 40)
    print(f"Prediction Complete!")
    print(f"Results saved to: {output_file}")
    print("=" * 40)

    # Show top 5 "Sieve Clearers" overall
    try:
        print("\nTop 5 Prospects by P_Sieve across all 10 years:")
        print(df_final.head(5))
    except UnicodeEncodeError:
        print(
            "\nTop 5 Prospects by P_Sieve across all 10 years (CSV saved, console print failed due to encoding)"
        )


if __name__ == "__main__":
    main()
