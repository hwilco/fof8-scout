import warnings

import numpy as np
import polars as pl
import pytest
from fof8_ml.orchestration.evaluator import (
    compute_cross_outcome_metrics,
    compute_regressor_oof_metrics,
    optimize_threshold,
)


def test_optimize_threshold_warns_and_falls_back_deterministically_when_constraint_infeasible():
    y_true = np.array([1, 1, 1, 0, 0])
    calibrated_probs = np.array([0.02, 0.03, 0.04, 0.01, 0.01])

    with pytest.warns(RuntimeWarning, match="No threshold satisfied min_positive_recall"):
        threshold, best_f1_bust = optimize_threshold(
            y_true,
            calibrated_probs,
            min_positive_recall=1.1,
        )

    assert threshold == pytest.approx(0.02)
    assert best_f1_bust == pytest.approx(1.0)


def test_optimize_threshold_no_warning_when_constraint_is_feasible():
    y_true = np.array([1, 1, 0, 0])
    calibrated_probs = np.array([0.9, 0.8, 0.2, 0.1])

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        threshold, _ = optimize_threshold(
            y_true,
            calibrated_probs,
            min_positive_recall=0.5,
        )

    assert len(recorded) == 0
    assert 0.01 <= threshold <= 0.99


def test_compute_regressor_oof_metrics_converts_log_space_to_raw_space():
    y_true = np.log1p(np.array([10.0, 20.0]))
    y_pred = np.log1p(np.array([13.0, 16.0]))

    metrics = compute_regressor_oof_metrics(y_true, y_pred, target_space="log")

    assert metrics["regressor_oof_rmse"] == pytest.approx(3.5355339)
    assert metrics["regressor_oof_mae"] == pytest.approx(3.5)


def test_compute_regressor_oof_metrics_uses_raw_space_directly():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([13.0, 16.0])

    metrics = compute_regressor_oof_metrics(
        y_true,
        y_pred,
        target_space="raw",
        draft_year=np.array([2020, 2021]),
    )

    assert metrics["regressor_oof_rmse"] == pytest.approx(3.5355339)
    assert metrics["regressor_oof_mae"] == pytest.approx(3.5)
    assert "regressor_mean_ndcg_at_128" in metrics


def test_compute_regressor_oof_metrics_clips_negative_targets_for_ranking_relevance():
    y_true = np.array([-1.0, 0.0, 2.0])
    y_pred = np.array([0.9, 0.8, 0.1])
    metrics = compute_regressor_oof_metrics(
        y_true,
        y_pred,
        target_space="raw",
        draft_year=np.array([2020, 2020, 2021]),
    )
    assert 0.0 <= metrics["regressor_mean_ndcg_at_32"] <= 1.0
    assert 0.0 <= metrics["regressor_mean_ndcg_at_128"] <= 1.0
    assert "regressor_draft_value_score" in metrics


def test_compute_cross_outcome_metrics_computes_available_families_and_skips_missing():
    y_score = np.array([0.9, 0.8, 0.4, 0.2])
    draft_year = np.array([2020, 2020, 2021, 2021])
    outcomes = pl.DataFrame(
        {
            "Career_Merit_Cap_Share": [10.0, 5.0, 0.0, 2.0],
            "Peak_Overall": [70.0, 65.0, 60.0, 58.0],
            "Career_Games_Played": [100.0, 80.0, 60.0, 40.0],
            "Economic_Success": [1, 1, 0, 1],
        }
    )
    metrics = compute_cross_outcome_metrics(y_score, outcomes, draft_year)

    assert "cross_econ_mean_ndcg_at_64" in metrics
    assert "cross_talent_mean_ndcg_at_64" in metrics
    assert "cross_longevity_mean_ndcg_at_64" in metrics
    assert "cross_bust_rate_at_32" in metrics
    assert metrics["cross_elite_available"] == 0.0
