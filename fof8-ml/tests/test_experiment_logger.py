from pathlib import Path

import numpy as np
import polars as pl
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.pipeline_types import (
    ClassifierResult,
    CVResult,
    PreparedData,
    TimelineInfo,
)
from omegaconf import OmegaConf


class _DummyModel:
    def __init__(self):
        self.logged = False

    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        self.logged = True
        return None

    def require_model(self):
        return {"dummy": True}

    def get_feature_importance(self):
        return [], np.array([])


def _prepared_data() -> PreparedData:
    timeline = TimelineInfo(
        initial_year=2020,
        final_sim_year=2024,
        valid_start_year=2020,
        valid_end_year=2024,
        train_year_range=[2020, 2022],
        val_year_range=[2023, 2023],
        test_year_range=[2024, 2024],
    )
    return PreparedData(
        X_train=pl.DataFrame({"feature": [1.0, 2.0]}),
        X_val=pl.DataFrame({"feature": [3.0]}),
        X_test=pl.DataFrame({"feature": [4.0]}),
        y_cls=np.array([1, 0]),
        y_cls_val=np.array([1]),
        y_cls_test=np.array([0]),
        y_reg=np.array([1.0, 2.0]),
        y_reg_val=np.array([3.0]),
        y_reg_test=np.array([4.0]),
        meta_train=pl.DataFrame({"Universe": ["A", "B"]}),
        meta_val=pl.DataFrame({"Universe": ["C"]}),
        meta_test=pl.DataFrame({"Universe": ["D"]}),
        timeline=timeline,
        metadata_columns=["Universe"],
        target_columns=["Economic_Success"],
    )


def _classifier_result(n_rows: int = 1) -> ClassifierResult:
    probs = np.full(n_rows, 0.7)
    preds = np.ones(n_rows, dtype=int)
    return ClassifierResult(
        cv_result=CVResult(
            oof_predictions=probs,
            best_iterations=[10],
            fold_metrics=[],
        ),
        calibrated_oof_probs=probs,
        raw_oof_probs=probs,
        optimal_threshold=0.5,
        final_predictions=preds,
        metrics={"classifier_val_pr_auc": 1.0},
        calibrator=object(),
    )


def test_log_classifier_results_uses_val_artifact_names(monkeypatch, tmp_path):
    logger = ExperimentLogger(exp_root=str(tmp_path), experiment_name="test")
    artifacts: list[str] = []
    captured_labels: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.mlflow.log_params", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.mlflow.log_metrics", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.mlflow.log_artifact",
        lambda path: artifacts.append(Path(path).name),
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.log_calibration_comparison",
        lambda *_a, eval_label="oof", **_k: captured_labels.append(f"cal:{eval_label}"),
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.log_confusion_matrix",
        lambda *_a, eval_label="oof", **_k: captured_labels.append(f"cm:{eval_label}"),
    )
    cfg = OmegaConf.create({"diagnostics": {"log_importance": False, "log_shap": False}})

    logger.log_classifier_results(
        _classifier_result(),
        _DummyModel(),
        _prepared_data(),
        cfg,
        quiet=False,
        eval_split_label="val",
        test_calibrated_probs=np.array([0.2]),
        test_raw_probs=np.array([0.2]),
        test_threshold=0.5,
    )

    assert "classifier_val_results.csv" in artifacts
    assert "classifier_test_results.csv" in artifacts
    assert captured_labels == ["cal:val", "cm:val"]


def test_log_classifier_results_keeps_oof_artifact_names(monkeypatch, tmp_path):
    logger = ExperimentLogger(exp_root=str(tmp_path), experiment_name="test")
    artifacts: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.mlflow.log_params", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.mlflow.log_metrics", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.mlflow.log_artifact",
        lambda path: artifacts.append(Path(path).name),
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.log_calibration_comparison",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.log_confusion_matrix",
        lambda *_a, **_k: None,
    )
    cfg = OmegaConf.create({"diagnostics": {"log_importance": False, "log_shap": False}})

    logger.log_classifier_results(
        _classifier_result(n_rows=2),
        _DummyModel(),
        _prepared_data(),
        cfg,
        quiet=False,
        eval_split_label="oof",
    )

    assert "classifier_oof_results.csv" in artifacts


def test_log_regressor_results_can_skip_remote_mlflow_model_logging(monkeypatch, tmp_path):
    logger = ExperimentLogger(exp_root=str(tmp_path), experiment_name="test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_logger.mlflow.log_metrics", lambda *_a, **_k: None
    )
    model = _DummyModel()
    cfg = OmegaConf.create(
        {
            "diagnostics": {"skip_mlflow_model_logging": True},
            "target": {"regressor_intensity": {"target_space": "raw"}},
        }
    )

    logger.log_regressor_results(
        {"regressor_val_draft_value_score": 1.0},
        model,
        pl.DataFrame({"feature": [1.0, 2.0]}),
        cfg,
        quiet=True,
    )

    assert model.logged is False
