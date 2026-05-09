from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import polars as pl
from fof8_ml.orchestration.classifier import run_classifier
from fof8_ml.orchestration.progress import (
    print_phase,
    print_test_phase,
    print_validation_run_start,
)
from fof8_ml.orchestration.regressor import run_regressor
from omegaconf import OmegaConf


def test_print_validation_run_start_reports_universe_and_row_counts(capsys):
    print_validation_run_start(
        role_label="classifier",
        meta_train=pl.DataFrame({"Universe": ["A", "A", "B"]}),
        meta_val=pl.DataFrame({"Universe": ["C"]}),
        meta_test=pl.DataFrame({"Universe": ["D", "E"]}),
        extra_detail="Extra detail.",
    )
    print_phase("Calibrating")
    print_test_phase("classifier", 12)

    out = capsys.readouterr().out

    assert "train=2 (3 rows)" in out
    assert "val=1 (1 rows)" in out
    assert "test=2 (2 rows)" in out
    assert "Extra detail." in out
    assert "[Calibrating]" in out
    assert "[Scoring held-out test universes for classifier (12 rows)]" in out


def test_classifier_uses_runtime_progress_setting_for_training_calls(monkeypatch):
    captured: dict[str, int] = {}

    class _DummyModel:
        def predict_proba(self, X):
            return np.full(len(X), 0.8)

        def get_best_iteration(self):
            return 12

    class _IdentityCalibrator:
        def fit(self, y_prob, y_true):
            return self

        def predict(self, y_prob):
            return y_prob

    def _train_with_validation(**kwargs):
        captured["validation"] = kwargs["interactive_progress_every"]
        return _DummyModel()

    def _train_final(**kwargs):
        captured["final"] = kwargs["interactive_progress_every"]
        return _DummyModel()

    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.train_model_with_validation",
        _train_with_validation,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.train_final_model",
        _train_final,
    )
    monkeypatch.setattr("fof8_ml.orchestration.classifier.BetaCalibrator", _IdentityCalibrator)
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.run_calibration_audit", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.optimize_threshold",
        lambda **_kwargs: (0.5, 0.0),
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.compute_classifier_final_metrics",
        lambda **_kwargs: {
            "classifier_oof_f1_bust": 1.0,
            "classifier_oof_recall_bust": 1.0,
            "classifier_oof_pr_auc": 1.0,
        },
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.joblib.dump", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.mlflow.log_param", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.mlflow.log_params", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.mlflow.log_metrics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.classifier.mlflow.log_artifact", lambda *_args, **_kwargs: None
    )

    cfg = OmegaConf.create(
        {
            "model": {"name": "catboost_classifier", "params": {}},
            "target": {
                "classifier_sieve": {
                    "target_col": "Economic_Success",
                    "min_positive_recall": 0.0,
                }
            },
            "seed": 42,
            "use_gpu": False,
            "runtime": {"catboost_progress_every": 17},
        }
    )
    data = SimpleNamespace(
        X_train=pl.DataFrame({"feature": [1.0, 2.0]}),
        X_val=pl.DataFrame({"feature": [3.0]}),
        X_test=pl.DataFrame({"feature": [4.0]}),
        y_cls=np.array([1, 0]),
        y_cls_val=np.array([1]),
        y_cls_test=np.array([1]),
        meta_train=pl.DataFrame({"Universe": ["A", "B"]}),
        meta_val=pl.DataFrame({"Universe": ["C"]}),
        meta_test=pl.DataFrame({"Universe": ["D"]}),
    )
    logger = SimpleNamespace(
        start_model_run=lambda *_args, **_kwargs: nullcontext(),
        log_model_params=lambda *_args, **_kwargs: None,
        log_classifier_results=lambda *_args, **_kwargs: None,
    )
    ctx = SimpleNamespace(
        cfg=cfg,
        data=data,
        logger=logger,
        sweep_context=SimpleNamespace(quiet=True),
    )

    run_classifier(ctx)

    assert captured["validation"] == 17
    assert captured["final"] == 17


def test_regressor_uses_runtime_progress_setting_for_training_calls(monkeypatch):
    captured: dict[str, int] = {}

    class _DummyModel:
        def predict(self, X):
            return np.full(len(X), 1.0)

        def get_best_iteration(self):
            return 9

    def _train_with_validation(**kwargs):
        captured["validation"] = kwargs["interactive_progress_every"]
        return _DummyModel()

    def _train_final(**kwargs):
        captured["final"] = kwargs["interactive_progress_every"]
        return _DummyModel()

    monkeypatch.setattr("fof8_ml.orchestration.regressor.get_model_family", lambda **_: "catboost")
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.train_model_with_validation",
        _train_with_validation,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.train_final_model",
        _train_final,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.compute_regressor_oof_metrics",
        lambda **_kwargs: {
            "regressor_oof_rmse": 1.0,
            "regressor_oof_mae": 1.0,
            "regressor_draft_value_score": 0.5,
        },
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.compute_cross_outcome_metrics",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_param", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.regressor.mlflow.log_metrics", lambda *_args, **_kwargs: None
    )

    cfg = OmegaConf.create(
        {
            "model": {"name": "catboost_tweedie_regressor", "params": {"loss_function": "RMSE"}},
            "target": {
                "regressor_intensity": {
                    "target_col": "Positive_Career_Merit_Cap_Share",
                    "target_space": "raw",
                }
            },
            "seed": 42,
            "use_gpu": False,
            "runtime": {"catboost_progress_every": 23},
        }
    )
    data = SimpleNamespace(
        X_train=pl.DataFrame({"feature": [1.0, 2.0]}),
        X_val=pl.DataFrame({"feature": [3.0]}),
        X_test=pl.DataFrame({"feature": [4.0]}),
        y_cls=np.array([1, 1]),
        y_cls_val=np.array([1]),
        y_cls_test=np.array([1]),
        y_reg=np.array([2.0, 3.0]),
        y_reg_val=np.array([4.0]),
        y_reg_test=np.array([5.0]),
        meta_train=pl.DataFrame({"Universe": ["A", "B"], "Year": [2020, 2021]}),
        meta_val=pl.DataFrame({"Universe": ["C"], "Year": [2022]}),
        meta_test=pl.DataFrame({"Universe": ["D"], "Year": [2023]}),
        outcomes_train=None,
        outcomes_val=None,
        outcomes_test=None,
    )
    logger = SimpleNamespace(
        start_model_run=lambda *_args, **_kwargs: nullcontext(),
        log_model_params=lambda *_args, **_kwargs: None,
        log_regressor_results=lambda *_args, **_kwargs: None,
    )
    ctx = SimpleNamespace(
        cfg=cfg,
        data=data,
        logger=logger,
        sweep_context=SimpleNamespace(quiet=True),
    )

    run_regressor(ctx)

    assert captured["validation"] == 23
    assert captured["final"] == 23
