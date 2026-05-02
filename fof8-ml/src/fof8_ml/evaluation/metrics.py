from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    auc,
    brier_score_loss,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_career_threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calculates comprehensive metrics for the career-threshold classification task."""
    y_pred = (y_prob >= threshold).astype(int)

    # 1. Curve-based metrics (ranking quality)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_true, y_prob)

    # 2. Point-based metrics (at threshold)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)
    zero_division = cast(Any, 0.0)
    prec = precision_score(y_true, y_pred, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, zero_division=zero_division)

    # 3. Calibration & Loss (probability quality)
    loss = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f0.5_score_at_0.5": float(f05),
        "precision_at_0.5": float(prec),
        "recall_at_0.5": float(rec),
        "log_loss": float(loss),
        "brier_score": float(brier),
        "positive_rate_at_0.5": float(y_pred.mean()),
    }
