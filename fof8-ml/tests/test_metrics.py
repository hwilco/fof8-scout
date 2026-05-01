import numpy as np
from fof8_ml.evaluation.metrics import calculate_career_threshold_metrics


def test_calculate_career_threshold_metrics():
    y_true = np.array([1, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.6, 0.2])

    metrics = calculate_career_threshold_metrics(y_true, y_prob)

    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "positive_rate_at_0.5" in metrics
    assert "f0.5_score_at_0.5" in metrics
