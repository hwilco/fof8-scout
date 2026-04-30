import logging
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import polars as pl
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def log_feature_importance(
    wrapper, stage_name: str, X: pl.DataFrame | None = None, log_shap: bool = False
):
    """
    Helper to generate and log feature importance plots to MLflow.

    Args:
        wrapper: The trained model wrapper (ModelWrapper).
        stage_name: Name of the pipeline stage.
        X: Optional feature data for SHAP values.
        log_shap: Boolean indicating whether to log SHAP plots.
    """
    import polars as pl

    # 1. Main Importance Plot
    feature_names, importances = wrapper.get_feature_importance()

    # Determine title suffix based on model type
    model_type = str(type(wrapper)).lower()
    if "catboost" in model_type:
        title_suffix = " (PredictionValuesChange)"
    elif "xgboost" in model_type:
        title_suffix = " (Gain)"
    else:
        title_suffix = " (Importance/Weight)"

    fi_df = pl.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort("Importance", descending=True)

    # For plotting, we only want the top N features
    fi_plot_df = fi_df.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Reverse for horizontal bar chart (top at the top)
    ax.barh(
        fi_plot_df.get_column("Feature").to_list()[::-1],
        fi_plot_df.get_column("Importance").to_list()[::-1],
        color="steelblue",
    )
    ax.set_title(f"{stage_name}{title_suffix}")
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_name_base = f"{stage_name.lower().replace(' ', '_')}"
        plot_path = os.path.join(tmpdir, f"{plot_name_base}_importance.png")
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        csv_path = os.path.join(tmpdir, f"{plot_name_base}_importance.csv")
        fi_df.write_csv(csv_path)
        mlflow.log_artifact(csv_path)

        # 2. SHAP Summary Plots (Global and Per-Position)
        if log_shap and X is not None:
            try:
                import shap

                # Preprocess X using the wrapper's transformation
                X_transformed = wrapper.transform(X)

                # SHAP often prefers Pandas or Numpy
                # For CatBoost, we MUST use the Pandas format it was trained on
                if "catboost" in model_type:
                    X_transformed_pd = X_transformed.to_pandas()
                else:
                    X_transformed_pd = X_transformed.to_pandas()

                explainer = shap.TreeExplainer(wrapper.model)

                # Global SHAP
                sample_size = min(1000, len(X_transformed_pd))
                X_sample = (
                    X_transformed_pd.sample(sample_size, random_state=42)
                    if len(X_transformed_pd) > sample_size
                    else X_transformed_pd
                )
                shap_values = explainer.shap_values(X_sample)

                fig_shap = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.title(f"{stage_name} - Global SHAP Influence")
                plt.tight_layout()

                shap_path = os.path.join(tmpdir, f"{plot_name_base}_shap_global.png")
                plt.savefig(shap_path)
                mlflow.log_artifact(shap_path)
                plt.close(fig_shap)

                # Per-Position SHAP
                # Note: We use the ORIGINAL X to find positions, but SHAP uses the TRANSFORMED X
                if "Position_Group" in X.columns:
                    positions = X.get_column("Position_Group").unique().to_list()
                    for pos in positions:
                        # Get indices for this position
                        pos_mask = X.get_column("Position_Group") == pos
                        X_pos_transformed = X_transformed.filter(pos_mask)

                        if len(X_pos_transformed) < 30:
                            continue

                        X_pos_pd = X_pos_transformed.to_pandas()
                        pos_sample_size = min(500, len(X_pos_pd))
                        X_pos_sample = (
                            X_pos_pd.sample(pos_sample_size, random_state=42)
                            if len(X_pos_pd) > pos_sample_size
                            else X_pos_pd
                        )

                        shap_pos = explainer.shap_values(X_pos_sample)

                        # Filter out all-NaN columns for cleaner plots
                        valid_cols = [
                            c for c in X_pos_sample.columns if not X_pos_sample[c].isna().all()
                        ]
                        if not valid_cols:
                            continue

                        X_pos_filtered = X_pos_sample[valid_cols]
                        # Correctly slice shap_pos if it's a 2D array
                        if isinstance(shap_values, list):  # Multi-class
                            # We take the first class for simplicity or handle as needed
                            cols_idx = [X_pos_sample.columns.get_loc(c) for c in valid_cols]
                            shap_pos_filtered = shap_pos[0][:, cols_idx]
                        else:
                            cols_idx = [X_pos_sample.columns.get_loc(c) for c in valid_cols]
                            shap_pos_filtered = shap_pos[:, cols_idx]

                        fig_pos = plt.figure(figsize=(12, 8))
                        shap.summary_plot(shap_pos_filtered, X_pos_filtered, show=False)
                        plt.title(f"{stage_name} - SHAP Influence ({pos})")
                        plt.tight_layout()

                        pos_path = os.path.join(
                            tmpdir, f"{plot_name_base}_shap_{str(pos).lower()}.png"
                        )
                        plt.savefig(pos_path)
                        mlflow.log_artifact(pos_path)
                        plt.close(fig_pos)

            except Exception as e:
                logging.warning(f"Could not generate SHAP plots for {stage_name}: {e}")

    plt.close(fig)


def log_confusion_matrix(y_true, y_pred, threshold: float):
    """
    Helper to generate and log confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        threshold: Classification threshold used.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Bust", "Hit"]).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"OOF Confusion Matrix (Threshold: {threshold:.3f})")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "oof_confusion_matrix.png")
        fig.savefig(path)
        mlflow.log_artifact(path)
    plt.close(fig)


def log_calibration_comparison(y_true, y_prob_raw, y_prob_cal, n_bins: int = 10):
    """
    Generates and logs a reliability diagram comparing raw and calibrated probabilities.

    Args:
        y_true: Ground truth binary labels.
        y_prob_raw: Uncalibrated probabilities.
        y_prob_cal: Calibrated probabilities.
        n_bins: Number of bins for the calibration curve.
    """
    from sklearn.calibration import calibration_curve

    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=n_bins)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_cal, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", label="Raw (Uncalibrated)", alpha=0.6)
    ax.plot(prob_pred_cal, prob_true_cal, "d-", label="Calibrated (Beta)", color="darkorange")

    ax.set_ylabel("Fraction of Positives (Actual)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")
    ax.set_title("Calibration Reliability Diagram (Pre vs Post)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "calibration_reliability_plot.png")
        fig.savefig(path)
        mlflow.log_artifact(path)

    plt.close(fig)
