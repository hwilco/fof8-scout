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


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute NDCG@k from relevance labels and ranking scores."""
    if k <= 0 or y_true.size == 0:
        return 0.0

    relevance = np.maximum(y_true.astype(float), 0.0)
    if not np.any(relevance > 0):
        return 0.0

    k_eff = min(k, relevance.size)
    order = np.argsort(-y_score)[:k_eff]
    ranked_rel = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    dcg = float(np.sum((np.power(2.0, ranked_rel) - 1.0) * discounts))

    ideal_rel = np.sort(relevance)[::-1][:k_eff]
    idcg = float(np.sum((np.power(2.0, ideal_rel) - 1.0) * discounts))
    return 0.0 if idcg <= 0 else float(dcg / idcg)


def mean_ndcg_by_group(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int
) -> float:
    """Compute mean NDCG@k over groups (e.g., draft years)."""
    if y_true.size == 0:
        return 0.0
    values = []
    for group in np.unique(groups):
        mask = groups == group
        values.append(ndcg_at_k(y_true[mask], y_score[mask], k))
    return float(np.mean(values)) if values else 0.0


def topk_weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute weighted MAE on the top-k by predicted value."""
    if k <= 0 or y_true.size == 0:
        return 0.0

    k_eff = min(k, y_true.size)
    order = np.argsort(-y_pred)[:k_eff]
    errors = np.abs(y_pred[order] - y_true[order])
    weights = np.arange(k_eff, 0, -1, dtype=float)
    return float(np.average(errors, weights=weights))


def topk_weighted_mae_normalized(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute weighted MAE normalized by weighted absolute target magnitude in top-k."""
    if k <= 0 or y_true.size == 0:
        return 0.0

    k_eff = min(k, y_true.size)
    order = np.argsort(-y_pred)[:k_eff]
    weights = np.arange(k_eff, 0, -1, dtype=float)
    scale = float(np.average(np.abs(y_true[order]), weights=weights))
    mae = topk_weighted_mae(y_true, y_pred, k_eff)
    return 0.0 if scale <= 1e-12 else float(mae / scale)


def topk_bias(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute weighted signed error on the top-k by predicted value."""
    if k <= 0 or y_true.size == 0:
        return 0.0

    k_eff = min(k, y_true.size)
    order = np.argsort(-y_pred)[:k_eff]
    weights = np.arange(k_eff, 0, -1, dtype=float)
    bias = np.average(y_pred[order] - y_true[order], weights=weights)
    return float(bias)


def calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute slope of linear calibration fit y_true ~ slope * y_pred + intercept."""
    if y_true.size == 0:
        return 0.0
    pred_var = float(np.var(y_pred))
    if pred_var <= 1e-12:
        return 0.0
    cov = float(np.cov(y_pred, y_true, ddof=0)[0, 1])
    return float(cov / pred_var)


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
