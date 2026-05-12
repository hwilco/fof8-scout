from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from fof8_ml.orchestration.pipeline_types import CVResult
from fof8_ml.orchestration.regressor import run_regressor
from omegaconf import OmegaConf


def _make_context(
    *,
    target_space: str,
    loss_function: str,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    meta_train: pl.DataFrame | None = None,
):
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "catboost_tweedie_regressor",
                "params": {"loss_function": loss_function},
            },
            "target": {
                "regressor_intensity": {
                    "target_col": "Positive_Career_Merit_Cap_Share",
                    "target_space": target_space,
                    "row_filter": "classifier_positive",
                }
            },
            "cv": {"n_folds": 2, "shuffle": True},
            "seed": 42,
            "use_gpu": False,
            "runtime": {"catboost_progress_every": 100},
        }
    )
    data = SimpleNamespace(
        X_train=pl.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        y_cls=y_cls,
        y_reg=y_reg,
        meta_train=meta_train
        if meta_train is not None
        else pl.DataFrame(
            {
                "Universe": ["A", "A", "B", "B"],
                "Year": [2020, 2020, 2020, 2020],
            }
        ),
        outcomes_train=pl.DataFrame(
            {
                "Positive_Career_Merit_Cap_Share": y_reg,
                "Career_Merit_Cap_Share": y_reg,
                "Peak_Overall": [50.0, 55.0, 60.0, 65.0],
                "Career_Games_Played": [16.0, 32.0, 48.0, 64.0],
                "Economic_Success": y_cls,
            }
        ),
    )
    logger = SimpleNamespace(
        start_model_run=lambda *_args, **_kwargs: nullcontext(),
        log_model_params=lambda *_args, **_kwargs: None,
        log_regressor_results=lambda *_args, **_kwargs: None,
    )
    sweep_context = SimpleNamespace(quiet=True)
    return SimpleNamespace(cfg=cfg, data=data, logger=logger, sweep_context=sweep_context)


def test_run_regressor_rejects_invalid_target_space():
    ctx = _make_context(
        target_space="invalid",
        loss_function="Tweedie",
        y_reg=np.array([1.0, 2.0, 3.0, 4.0]),
        y_cls=np.array([1, 1, 1, 1]),
    )
    with pytest.raises(ValueError, match="Unsupported regressor target_space"):
        run_regressor(ctx)


def test_run_regressor_requires_raw_space_for_tweedie():
    ctx = _make_context(
        target_space="log",
        loss_function="Tweedie",
        y_reg=np.array([1.0, 2.0, 3.0, 4.0]),
        y_cls=np.array([1, 1, 1, 1]),
    )
    with pytest.raises(ValueError, match="requires target.regressor_intensity.target_space='raw'"):
        run_regressor(ctx)


def test_run_regressor_rejects_negative_targets_for_tweedie():
    ctx = _make_context(
        target_space="raw",
        loss_function="Tweedie",
        y_reg=np.array([1.0, -0.5, 3.0, 4.0]),
        y_cls=np.array([1, 1, 1, 1]),
    )
    with pytest.raises(ValueError, match="requires non-negative regressor targets"):
        run_regressor(ctx)


def test_run_regressor_uses_configured_target_space(monkeypatch):
    captured = {}

    def fake_run_cv_regressor(*, X, y, target_space, **_kwargs):
        captured["y"] = y
        captured["target_space"] = target_space
        return CVResult(
            oof_predictions=np.array([0.1, 0.2, 0.3, 0.4]),
            best_iterations=[10, 12],
            fold_metrics=[{"rmse": 1.0, "mae": 1.0}, {"rmse": 1.2, "mae": 1.1}],
        )

    monkeypatch.setattr("fof8_ml.orchestration.regressor.get_model_family", lambda **_: "catboost")
    monkeypatch.setattr("fof8_ml.orchestration.regressor.run_cv_regressor", fake_run_cv_regressor)
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.compute_regressor_oof_metrics",
        lambda **_kwargs: {
            "regressor_oof_rmse": 1.0,
            "regressor_oof_mae": 1.0,
            "regressor_oof_top32_target_capture_ratio": 0.8,
            "regressor_oof_top64_target_capture_ratio": 0.9,
            "regressor_oof_mean_ndcg_at_32": 0.7,
            "regressor_oof_mean_ndcg_at_64": 0.75,
            "regressor_oof_draft_value_score": 0.75,
        },
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.train_final_model", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_param", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_metrics", lambda *_args, **_kwargs: None
    )

    y_reg = np.array([2.0, 3.0, 4.0, 5.0])
    y_cls = np.array([1, 1, 1, 1])
    ctx = _make_context(target_space="log", loss_function="RMSE", y_reg=y_reg, y_cls=y_cls)
    run_regressor(ctx)

    assert captured["target_space"] == "log"
    assert np.allclose(captured["y"], np.log1p(y_reg))


def test_run_regressor_can_filter_on_observed_regression_target_instead_of_classifier(monkeypatch):
    captured = {}

    def fake_run_cv_regressor(*, X, y, target_space, **_kwargs):
        captured["rows"] = len(X)
        captured["y"] = y
        captured["target_space"] = target_space
        return CVResult(
            oof_predictions=np.array([0.1, 0.2, 0.3]),
            best_iterations=[10, 12],
            fold_metrics=[{"rmse": 1.0, "mae": 1.0}, {"rmse": 1.2, "mae": 1.1}],
        )

    monkeypatch.setattr("fof8_ml.orchestration.regressor.get_model_family", lambda **_: "catboost")
    monkeypatch.setattr("fof8_ml.orchestration.regressor.run_cv_regressor", fake_run_cv_regressor)
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.compute_regressor_oof_metrics",
        lambda **_kwargs: {
            "regressor_oof_rmse": 1.0,
            "regressor_oof_mae": 1.0,
            "regressor_oof_top32_target_capture_ratio": 0.8,
            "regressor_oof_top64_target_capture_ratio": 0.9,
            "regressor_oof_mean_ndcg_at_32": 0.7,
            "regressor_oof_mean_ndcg_at_64": 0.75,
            "regressor_oof_draft_value_score": 0.75,
        },
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.train_final_model", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_param", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_metrics", lambda *_args, **_kwargs: None
    )

    ctx = _make_context(
        target_space="raw",
        loss_function="RMSE",
        y_reg=np.array([2.0, np.nan, 4.0, 5.0]),
        y_cls=np.array([0, 0, 0, 0]),
    )
    ctx.cfg.target.regressor_intensity.row_filter = "regression_target_observed"
    run_regressor(ctx)

    assert captured["target_space"] == "raw"
    assert captured["rows"] == 3
    assert np.allclose(captured["y"], np.array([2.0, 4.0, 5.0]))


def test_run_regressor_requires_universe_and_year_for_draft_aware_metrics():
    ctx = _make_context(
        target_space="raw",
        loss_function="RMSE",
        y_reg=np.array([1.0, 2.0, 3.0, 4.0]),
        y_cls=np.array([1, 1, 1, 1]),
        meta_train=pl.DataFrame({"Year": [2020, 2020, 2021, 2021]}),
    )

    with pytest.raises(ValueError, match="must include Universe and Year"):
        run_regressor(ctx)


def test_run_regressor_passes_one_group_per_universe_year(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    def fake_run_cv_regressor(**_kwargs):
        return CVResult(
            oof_predictions=np.array([0.1, 0.2, 0.3, 0.4]),
            best_iterations=[10, 12],
            fold_metrics=[{"rmse": 1.0, "mae": 1.0}, {"rmse": 1.2, "mae": 1.1}],
        )

    def fake_compute_regressor_oof_metrics(**kwargs):
        captured["regressor_group"] = kwargs["draft_group"]
        return {
            "regressor_oof_rmse": 1.0,
            "regressor_oof_mae": 1.0,
            "regressor_oof_top32_target_capture_ratio": 0.8,
            "regressor_oof_top64_target_capture_ratio": 0.9,
            "regressor_oof_mean_ndcg_at_32": 0.7,
            "regressor_oof_mean_ndcg_at_64": 0.75,
            "regressor_oof_draft_value_score": 0.75,
        }

    def fake_compute_cross_outcome_metrics(**kwargs):
        captured["cross_group"] = kwargs["draft_group"]
        return {}

    monkeypatch.setattr("fof8_ml.orchestration.regressor.get_model_family", lambda **_: "catboost")
    monkeypatch.setattr("fof8_ml.orchestration.regressor.run_cv_regressor", fake_run_cv_regressor)
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.compute_regressor_oof_metrics",
        fake_compute_regressor_oof_metrics,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.compute_cross_outcome_metrics",
        fake_compute_cross_outcome_metrics,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.train_final_model", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_param", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_metrics", lambda *_args, **_kwargs: None
    )

    ctx = _make_context(
        target_space="raw",
        loss_function="RMSE",
        y_reg=np.array([2.0, 3.0, 4.0, 5.0]),
        y_cls=np.array([1, 1, 1, 1]),
        meta_train=pl.DataFrame(
            {
                "Universe": ["A", "A", "B", "B"],
                "Year": [2020, 2020, 2020, 2020],
            }
        ),
    )

    run_regressor(ctx)

    expected = np.array(["A:2020", "A:2020", "B:2020", "B:2020"], dtype=object)
    assert np.array_equal(captured["regressor_group"], expected)
    assert np.array_equal(captured["cross_group"], expected)
