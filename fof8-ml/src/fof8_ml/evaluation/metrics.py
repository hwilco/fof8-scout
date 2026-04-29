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


def calculate_survival_metrics(y_true, y_prob, threshold=0.5):
    """Calculates comprehensive metrics for the survival task."""
    y_pred = (y_prob >= threshold).astype(int)

    # 1. Curve-based metrics (ranking quality)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_true, y_prob)

    # 2. Point-based metrics (at threshold)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # 3. Calibration & Loss (probability quality)
    loss = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f0.5_score_at_0.5": f05,
        "precision_at_0.5": prec,
        "recall_at_0.5": rec,
        "log_loss": loss,
        "brier_score": brier,
        "positive_rate_at_0.5": float(y_pred.mean()),
    }
