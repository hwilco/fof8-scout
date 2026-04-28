import argparse
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from fof8_ml.data.dataset import build_survival_dataset
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow Run ID to load")
    parser.add_argument("--output", default="prob_histogram.png", help="Output filename")
    args = parser.parse_args()

    # 1. Connect to MLflow (Point to the DB in the root)
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")

    # 2. Get Run Info to see which model type it was
    run = mlflow.get_run(args.run_id)
    model_name = run.data.params.get("model.name", "catboost")  # Default to catboost if not found
    is_catboost = "catboost" in model_name.lower()

    print(f"Loading {model_name} model from run {args.run_id}...")

    # 3. Load Model
    model_uri = f"runs:/{args.run_id}/model"
    if is_catboost:
        model = mlflow.catboost.load_model(model_uri)
    else:
        model = mlflow.xgboost.load_model(model_uri)

    # 4. Load Data (Using defaults from your last session)
    X, y = build_survival_dataset(
        "../fof8-gen/data/raw",
        "DRAFT003",
        [2021, 2100],
        2144,
        5,
        target_column="Number_of_Seasons",
        positions=["QB"],
    )

    # 5. Predict Probs
    print("Generating predictions...")
    if is_catboost:
        X_pd = X.to_pandas()
        probs = model.predict_proba(X_pd)[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    # 6. Plotting
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for Seaborn
    plot_df = pl.DataFrame({"Probability": probs, "Actual": y}).to_pandas()

    # Plot overlapping histograms
    sns.histplot(data=plot_df, x="Probability", hue="Actual", bins=50, kde=True, element="step")

    plt.title(f"Probability Distribution: {model_name}\nRun: {args.run_id}")
    plt.xlabel("Predicted Probability of Survival")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)

    plt.savefig(args.output)
    print(f"Done! Histogram saved to {args.output}")


if __name__ == "__main__":
    main()
