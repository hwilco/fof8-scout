import numpy as np
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
        best_threshold = thresholds[0]
        final_preds = (calibrated_probs >= best_threshold).astype(int)
        best_f1_0 = f1_score(y_true, final_preds, pos_label=0)

    return float(best_threshold), float(best_f1_0)


def compute_stage1_final_metrics(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute all Stage 1 final metrics (bust precision, hit recall, PR-AUC, etc.)."""
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
        "s1_oof_busts_filtered": float(busts_filtered),
        "s1_oof_hit_recall": float(hit_recall),
        "s1_oof_f1_bust": float(f1_bust),
        "s1_oof_precision_bust": float(bust_precision),
        "s1_oof_recall_bust": float(bust_recall),
        "s1_oof_pr_auc": float(pr_auc),
        "s1_oof_roc_auc": float(roc_auc),
    }


def compute_stage2_oof_metrics(
    y_true_log: np.ndarray,
    oof_predictions_log: np.ndarray,
) -> dict[str, float]:
    """Compute Stage 2 OOF RMSE and MAE in original (expm1) space.

    Args:
        y_true_log: Ground truth continuous labels in log1p space.
        oof_predictions_log: Predicted continuous labels in log1p space.
    """
    y_real = np.expm1(y_true_log)
    y_pred = np.expm1(oof_predictions_log)

    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    mae = float(mean_absolute_error(y_real, y_pred))

    return {
        "s2_oof_rmse": rmse,
        "s2_oof_mae": mae,
    }
