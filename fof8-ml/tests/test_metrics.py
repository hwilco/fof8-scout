import numpy as np
import pytest
from fof8_ml.evaluation.metrics import (
    calculate_career_threshold_metrics,
    mean_ndcg_by_group,
    ndcg_at_k,
    topk_weighted_mae_normalized,
)


def test_calculate_career_threshold_metrics():
    y_true = np.array([1, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.6, 0.2])

    metrics = calculate_career_threshold_metrics(y_true, y_prob)

    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "positive_rate_at_0.5" in metrics
    assert "f0.5_score_at_0.5" in metrics


def test_ndcg_at_k_returns_zero_for_all_zero_relevance():
    y_true = np.array([0.0, 0.0, 0.0])
    y_score = np.array([0.9, 0.5, 0.1])
    assert ndcg_at_k(y_true, y_score, k=3) == 0.0


def test_mean_ndcg_by_group_handles_fewer_than_k():
    y_true = np.array([1.0, 0.0, 2.0])
    y_score = np.array([0.8, 0.7, 0.6])
    groups = np.array([2020, 2020, 2021])
    metric = mean_ndcg_by_group(y_true, y_score, groups, k=64)
    assert 0.0 <= metric <= 1.0


def test_topk_weighted_mae_normalized_handles_zero_scale():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.2, 0.1, 0.3])
    assert topk_weighted_mae_normalized(y_true, y_pred, k=2) == pytest.approx(0.0)
