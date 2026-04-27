import hydra
import mlflow
import polars as pl
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import os
import pandas as pd

from fof8_core.loader import FOF8Loader
from fof8_core.features import get_draft_class


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Define a stable root directory for the experimentation package (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Set the tracking URI to be stable within the experimentation package
    db_path = os.path.join(exp_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(cfg.experiment_name)

    # In a real scenario, you'd specify the run_id of your best pipeline run
    # For now, let's fetch the latest run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(cfg.experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'Pipeline_%'",
    )

    if not runs:
        print("No Pipeline runs found in MLflow. Please train the model first.")
        return

    latest_run = runs[0]
    parent_run_id = latest_run.info.run_id

    # Find nested runs
    nested_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    stage1_run = next(
        (r for r in nested_runs if r.data.tags.get("mlflow.runName") == "Stage1_Sieve_Classifier"),
        None,
    )
    stage2_run = next(
        (
            r
            for r in nested_runs
            if r.data.tags.get("mlflow.runName") == "Stage2_Intensity_Regressor"
        ),
        None,
    )

    if not stage1_run or not stage2_run:
        print("Could not find nested runs for Stage 1 or Stage 2.")
        return

    print(f"Loading Stage 1 Model from Run: {stage1_run.info.run_id}")
    stage1_model_uri = f"runs:/{stage1_run.info.run_id}/stage1_model"

    print(f"Loading Stage 2 Model from Run: {stage2_run.info.run_id}")
    stage2_model_uri = f"runs:/{stage2_run.info.run_id}/stage2_model"

    absolute_raw_path = to_absolute_path(cfg.data.raw_path)
    loader = FOF8Loader(base_path=absolute_raw_path, league_name=cfg.data.league_name)

    # Get active team ID for coach bias
    active_team_id = loader.get_active_team_id()

    # We need a year to inference on. Let's take the first year of year_range as default,
    # or ideally passed as an argument.
    inference_year = cfg.data.year_range[0]
    print(f"Generating draft board for year: {inference_year}")

    # 1. Fetch Features using the core library
    df_features = get_draft_class(loader, inference_year, active_team_id=active_team_id)

    # Fetch Age
    with pl.StringCache():
        df_info = (
            loader.scan_file("player_information.csv", year=2144)
            .select(["Player_ID", "Draft_Year", "Year_Born"])
            .collect()
        )

    df_features = (
        df_features.join(df_info, on="Player_ID", how="left")
        .with_columns((pl.col("Draft_Year") - pl.col("Year_Born")).alias("Age"))
        .drop(["Draft_Year", "Year_Born"])
    )

    # Need to apply the same preprocessing as training
    df_model = df_features.drop(["Player_ID", "Year", "First_Name", "Last_Name"])

    # Categorical logic
    X_pd = df_model.to_pandas()
    if "College" in X_pd.columns:
        X_pd["College"] = "Other"

    cat_cols = X_pd.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X_pd[col] = X_pd[col].astype("category")

    # Predict Sieve (Probabilities)
    if "catboost" in cfg.stage1_model.name.lower():
        s1_native = mlflow.catboost.load_model(stage1_model_uri)
    else:
        s1_native = mlflow.xgboost.load_model(stage1_model_uri)
    p_sieve = s1_native.predict_proba(X_pd)[:, 1]

    # Predict Expected DGO
    if "catboost" in cfg.stage2_model.name.lower():
        s2_native = mlflow.catboost.load_model(stage2_model_uri)
        expected_dgo_raw = s2_native.predict(X_pd)
    elif "sklearn" in cfg.stage2_model.name.lower():
        s2_native = mlflow.sklearn.load_model(stage2_model_uri)
        # GLMs need OHE, scaling, and exact column matching
        X_sk = pd.get_dummies(X_pd, drop_first=True)

        # Download artifacts to ensure alignment
        client = mlflow.tracking.MlflowClient()
        feature_file = client.download_artifacts(stage2_run.info.run_id, "stage2_features.txt")
        scaler_file = client.download_artifacts(stage2_run.info.run_id, "stage2_scaler.joblib")

        import joblib

        scaler = joblib.load(scaler_file)

        with open(feature_file, "r") as f:
            expected_features = [line.strip() for line in f.readlines()]

        X_sk = X_sk.reindex(columns=expected_features, fill_value=0)

        # Apply scaling
        X_sk_scaled = scaler.transform(X_sk)
        X_sk = pd.DataFrame(X_sk_scaled, columns=X_sk.columns, index=X_sk.index)

        expected_dgo_raw = s2_native.predict(X_sk)
    else:
        s2_native = mlflow.xgboost.load_model(stage2_model_uri)
        expected_dgo_raw = s2_native.predict(X_pd)

    expected_dgo = np.expm1(expected_dgo_raw)

    df_results = df_features.select(
        ["Player_ID", "First_Name", "Last_Name", "Position_Group"]
    ).with_columns([pl.Series("P_Sieve", p_sieve), pl.Series("Expected_DGO", expected_dgo)])

    df_results = df_results.with_columns(
        (pl.col("P_Sieve") * pl.col("Expected_DGO")).alias("Universal_EDV")
    )

    df_results = df_results.with_columns(
        (
            pl.col("Universal_EDV").rank(descending=False).over("Position_Group")
            / pl.len().over("Position_Group")
        ).alias("Positional_Tp")
    )

    df_results = df_results.sort("Universal_EDV", descending=True)

    output_path = to_absolute_path(f"draft_board_predictions_{inference_year}.csv")
    df_results.write_csv(output_path)
    print(f"Draft board saved to {output_path}")


if __name__ == "__main__":
    main()
