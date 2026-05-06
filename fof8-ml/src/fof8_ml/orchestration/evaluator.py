import warnings

import numpy as np
import polars as pl
from fof8_ml.evaluation.metrics import (
    calibration_slope,
    mean_ndcg_by_group,
    topk_bias,
    topk_weighted_mae,
    topk_weighted_mae_normalized,
)
from sklearn.metrics import (
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def optimize_threshold(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    min_positive_recall: float = 0.0,
) -> tuple[float, float]:
    """Find threshold maximizing F1(bust) subject to min recall constraint.

    Args:
        y_true: Ground truth binary labels.
        calibrated_probs: Predicted probabilities.
        min_positive_recall: Minimum required recall for the positive class.

    Returns:
        (optimal_threshold, best_f1_bust)
    """
    best_threshold = 0.5
    best_f1_0 = -1.0

    thresholds = np.linspace(0.01, 0.99, 99)
    for thresh in thresholds:
        current_preds = (calibrated_probs >= thresh).astype(int)
        f1_0 = f1_score(y_true, current_preds, pos_label=0)
        recall_1 = recall_score(y_true, current_preds)

        if recall_1 >= min_positive_recall:
            if f1_0 > best_f1_0:
                best_f1_0 = f1_0
                best_threshold = thresh

    if best_f1_0 == -1.0:
        # No threshold satisfies the requested minimum positive recall.
        # Fall back deterministically to the threshold that yields maximum
        # positive recall, then best bust F1, then the lowest threshold.
        candidates: list[tuple[float, float, float]] = []
        for thresh in thresholds:
            current_preds = (calibrated_probs >= thresh).astype(int)
            recall_1 = recall_score(y_true, current_preds)
            f1_0 = f1_score(y_true, current_preds, pos_label=0)
            candidates.append((recall_1, f1_0, thresh))

        best_recall, best_f1_0, best_threshold = max(
            candidates, key=lambda item: (item[0], item[1], -item[2])
        )
        warnings.warn(
            (
                "No threshold satisfied min_positive_recall="
                f"{min_positive_recall:.4f}; using deterministic fallback threshold="
                f"{best_threshold:.2f} with achieved_recall={best_recall:.4f}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    return float(best_threshold), float(best_f1_0)


def compute_classifier_final_metrics(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute all classifier final metrics (bust precision, hit recall, PR-AUC, etc.)."""
    final_preds = (calibrated_probs >= threshold).astype(int)

    busts_filtered = np.sum((y_true == 0) & (final_preds == 0))
    hit_recall = recall_score(y_true, final_preds)
    bust_precision = precision_score(y_true, final_preds, pos_label=0)
    bust_recall = recall_score(y_true, final_preds, pos_label=0)

    p, r, _ = precision_recall_curve(y_true, calibrated_probs)
    pr_auc = auc(r, p)
    roc_auc = roc_auc_score(y_true, calibrated_probs)

    # Compute F1 for busts as well for convenience
    f1_bust = f1_score(y_true, final_preds, pos_label=0)

    return {
        "classifier_oof_busts_filtered": float(busts_filtered),
        "classifier_oof_hit_recall": float(hit_recall),
        "classifier_oof_f1_bust": float(f1_bust),
        "classifier_oof_precision_bust": float(bust_precision),
        "classifier_oof_recall_bust": float(bust_recall),
        "classifier_oof_pr_auc": float(pr_auc),
        "classifier_oof_roc_auc": float(roc_auc),
    }


def compute_regressor_oof_metrics(
    y_true: np.ndarray,
    oof_predictions: np.ndarray,
    target_space: str = "log",
    draft_year: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute regressor OOF RMSE and MAE in original target space.

    Args:
        y_true: Ground truth continuous labels in `target_space`.
        oof_predictions: Predicted continuous labels in `target_space`.
        target_space: Either "log" for log1p labels/predictions or "raw" for
            labels/predictions already in original target units.
    """
    if target_space == "log":
        y_real = np.expm1(y_true)
        y_pred = np.expm1(oof_predictions)
    elif target_space == "raw":
        y_real = y_true
        y_pred = np.maximum(oof_predictions, 0)
    else:
        raise ValueError(f"Unsupported target_space '{target_space}'. Expected 'log' or 'raw'.")

    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    mae = float(mean_absolute_error(y_real, y_pred))
    metrics = {
        "regressor_oof_rmse": rmse,
        "regressor_oof_mae": mae,
    }

    years = draft_year if draft_year is not None else np.zeros_like(y_real, dtype=int)
    metrics.update(
        {
            "regressor_mean_ndcg_at_32": mean_ndcg_by_group(y_real, y_pred, years, k=32),
            "regressor_mean_ndcg_at_64": mean_ndcg_by_group(y_real, y_pred, years, k=64),
            "regressor_mean_ndcg_at_128": mean_ndcg_by_group(y_real, y_pred, years, k=128),
            "regressor_top64_weighted_mae": topk_weighted_mae(y_real, y_pred, k=64),
            "regressor_top64_weighted_mae_normalized": topk_weighted_mae_normalized(
                y_real, y_pred, k=64
            ),
            "regressor_top64_bias": topk_bias(y_real, y_pred, k=64),
            "regressor_top64_calibration_slope": calibration_slope(y_real, y_pred),
            "regressor_rmse_positive": rmse,
            "regressor_mae_positive": mae,
        }
    )
    metrics["regressor_draft_value_score"] = float(
        metrics["regressor_mean_ndcg_at_64"]
        - 0.25 * metrics["regressor_top64_weighted_mae_normalized"]
    )
    return metrics


def compute_cross_outcome_metrics(
    y_score: np.ndarray,
    outcome_columns: pl.DataFrame | None,
    draft_year: np.ndarray,
) -> dict[str, float]:
    """Evaluate one ranked board against multiple outcome families."""
    if outcome_columns is None:
        return {"cross_outcomes_available": 0.0}

    def _first_available(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in outcome_columns.columns:
                return col
        return None

    def _topk_actual_value_by_group(y_val: np.ndarray, k: int) -> float:
        values: list[float] = []
        for group in np.unique(draft_year):
            mask = draft_year == group
            if not np.any(mask):
                continue
            scores_g = y_score[mask]
            y_g = y_val[mask]
            k_eff = min(k, y_g.size)
            order = np.argsort(-scores_g)[:k_eff]
            values.append(float(np.sum(y_g[order])))
        return float(np.mean(values)) if values else 0.0

    def _precision_at_k_by_group(y_binary: np.ndarray, k: int) -> float:
        vals: list[float] = []
        for group in np.unique(draft_year):
            mask = draft_year == group
            if not np.any(mask):
                continue
            scores_g = y_score[mask]
            y_g = y_binary[mask]
            k_eff = min(k, y_g.size)
            order = np.argsort(-scores_g)[:k_eff]
            vals.append(float(np.mean(y_g[order])))
        return float(np.mean(vals)) if vals else 0.0

    def _bust_rate_at_k_by_group(y_success: np.ndarray, k: int) -> float:
        vals: list[float] = []
        for group in np.unique(draft_year):
            mask = draft_year == group
            if not np.any(mask):
                continue
            scores_g = y_score[mask]
            y_g = y_success[mask]
            k_eff = min(k, y_g.size)
            order = np.argsort(-scores_g)[:k_eff]
            vals.append(float(np.mean(1.0 - y_g[order])))
        return float(np.mean(vals)) if vals else 0.0

    metrics: dict[str, float] = {"cross_outcomes_available": 1.0}

    econ_col = _first_available(
        ["Positive_Career_Merit_Cap_Share", "Career_Merit_Cap_Share", "Positive_DPO"]
    )
    if econ_col is not None:
        y_econ = outcome_columns[econ_col].to_numpy()
        metrics["cross_econ_mean_ndcg_at_64"] = mean_ndcg_by_group(y_econ, y_score, draft_year, k=64)
        metrics["cross_econ_top64_actual_value"] = _topk_actual_value_by_group(y_econ, k=64)
    else:
        metrics["cross_econ_available"] = 0.0

    talent_col = _first_available(["Top3_Mean_Current_Overall", "Peak_Overall"])
    if talent_col is not None:
        y_talent = outcome_columns[talent_col].to_numpy()
        metrics["cross_talent_mean_ndcg_at_64"] = mean_ndcg_by_group(
            y_talent, y_score, draft_year, k=64
        )
    else:
        metrics["cross_talent_available"] = 0.0

    longevity_col = _first_available(["Career_Games_Played"])
    if longevity_col is not None:
        y_longevity = outcome_columns[longevity_col].to_numpy()
        metrics["cross_longevity_mean_ndcg_at_64"] = mean_ndcg_by_group(
            y_longevity, y_score, draft_year, k=64
        )
    else:
        metrics["cross_longevity_available"] = 0.0

    elite_col = _first_available(["Hall_of_Fame_Flag", "HOF_Points", "Award_Count"])
    if elite_col is not None:
        y_elite_raw = outcome_columns[elite_col].to_numpy()
        y_elite = (y_elite_raw > 0).astype(float)
        metrics["cross_elite_precision_at_64"] = _precision_at_k_by_group(y_elite, k=64)
    else:
        metrics["cross_elite_available"] = 0.0

    success_col = _first_available(["Economic_Success", "Positive_Career_Merit_Cap_Share"])
    if success_col is not None:
        y_success_raw = outcome_columns[success_col].to_numpy()
        y_success = (y_success_raw > 0).astype(float)
        metrics["cross_bust_rate_at_32"] = _bust_rate_at_k_by_group(y_success, k=32)
    else:
        metrics["cross_bust_available"] = 0.0

    return metrics
