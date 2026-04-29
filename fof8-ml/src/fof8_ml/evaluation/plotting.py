import logging
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def log_feature_importance(model, feature_names, stage_name, is_catboost, X=None, log_shap=False):
    """
    Helper to generate and log feature importance plots to MLflow.

    Args:
        model: The trained model wrapper.
        feature_names: List of feature names.
        stage_name: Name of the pipeline stage.
        is_catboost: Boolean indicating if model is CatBoost.
        X: Optional feature data for SHAP values.
        log_shap: Boolean indicating whether to log SHAP plots.
    """

    # 1. Main Importance Plot (PredictionValuesChange or Weights)
    if is_catboost:
        importances = model.get_feature_importance()
        title_suffix = " (PredictionValuesChange)"
    else:
        # Works for XGBoost and Sklearn
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            title_suffix = " (Gain)"
        else:
            importances = np.abs(getattr(model, "coef_", np.zeros(len(feature_names))))
            title_suffix = " (Normalized Weights)"

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    # For plotting, we only want the top N features
    fi_plot_df = fi_df.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi_plot_df["Feature"][::-1], fi_plot_df["Importance"][::-1], color="steelblue")
    ax.set_title(f"{stage_name}{title_suffix}")
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_name_base = f"{stage_name.lower().replace(' ', '_')}"
        plot_path = os.path.join(tmpdir, f"{plot_name_base}_importance.png")
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        csv_path = os.path.join(tmpdir, f"{plot_name_base}_importance.csv")
        fi_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # 2. SHAP Summary Plots (Global and Per-Position)
        if log_shap and X is not None and (is_catboost or "XGB" in str(type(model))):
            try:
                import shap

                explainer = shap.TreeExplainer(model)

                # Global SHAP
                X_sample = X.sample(min(1000, len(X)), random_state=42) if len(X) > 1000 else X
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
                if "Position_Group" in X.columns:
                    positions = X["Position_Group"].unique()
                    for pos in positions:
                        X_pos = X[X["Position_Group"] == pos]
                        if len(X_pos) < 30:  # Minimum sample threshold for meaningful SHAP
                            continue

                        X_pos_sample = (
                            X_pos.sample(min(500, len(X_pos)), random_state=42)
                            if len(X_pos) > 500
                            else X_pos
                        )

                        # Calculate SHAP values on the FULL feature set
                        # (required for model compatibility)
                        shap_pos = explainer.shap_values(X_pos_sample)

                        # Identify columns that are NOT all-NaN to avoid plotting warnings
                        valid_cols_idx = [
                            i
                            for i, c in enumerate(X_pos_sample.columns)
                            if not X_pos_sample[c].isna().all()
                        ]

                        if not valid_cols_idx:
                            continue

                        # Filter both the SHAP values and the dataframe for the plot
                        shap_pos_filtered = shap_pos[:, valid_cols_idx]
                        X_pos_filtered = X_pos_sample[
                            [X_pos_sample.columns[i] for i in valid_cols_idx]
                        ]

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
