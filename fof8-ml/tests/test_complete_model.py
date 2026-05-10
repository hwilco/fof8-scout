import numpy as np
import polars as pl
import pytest
from fof8_ml.data.schema import FeatureSchema
from fof8_ml.evaluation.complete_model import (
    LoadedClassifierBundle,
    LoadedRegressorBundle,
    _load_role_bundle,
    evaluate_complete_model,
    evaluate_complete_model_by_slice,
    predict_complete_model,
)


class _FakeClassifierModel:
    def predict_proba(self, X):
        probs = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1.0 - probs, probs])


class _FakeRegressorModel:
    def predict(self, X):
        return np.asarray(X, dtype=float).reshape(-1)


class _IdentityCalibrator:
    def predict(self, probs):
        return probs


def test_predict_complete_model_applies_per_role_schemas_and_multiplies_outputs():
    X_full = pl.DataFrame(
        {
            "classifier_feature": [0.2, 0.8],
            "regressor_feature": [10.0, -5.0],
            "unused": [1, 2],
        }
    )
    classifier_bundle = LoadedClassifierBundle(
        role="classifier",
        run_id="cls",
        family="catboost",
        model=_FakeClassifierModel(),
        schema=FeatureSchema(
            feature_columns=["classifier_feature"],
            categorical_columns=[],
            excluded_metadata_columns=[],
            excluded_target_columns=[],
        ),
        threshold=0.5,
        calibrator=_IdentityCalibrator(),
    )
    regressor_bundle = LoadedRegressorBundle(
        role="regressor",
        run_id="reg",
        family="catboost",
        model=_FakeRegressorModel(),
        schema=FeatureSchema(
            feature_columns=["regressor_feature"],
            categorical_columns=[],
            excluded_metadata_columns=[],
            excluded_target_columns=[],
        ),
        target_space="raw",
    )

    predictions = predict_complete_model(X_full, classifier_bundle, regressor_bundle)

    assert np.allclose(predictions["classifier_probability"], [0.2, 0.8])
    assert np.allclose(predictions["regressor_prediction"], [10.0, 0.0])
    assert np.allclose(predictions["complete_prediction"], [2.0, 0.0])


def test_evaluate_complete_model_logs_core_and_cross_outcome_metrics():
    y_true = np.array([10.0, 0.0, 5.0, 0.0])
    y_pred = np.array([0.9, 0.2, 0.8, 0.1])
    draft_group = np.array(["A:2020", "A:2020", "B:2020", "B:2020"], dtype=object)
    outcomes = pl.DataFrame(
        {
            "Positive_Career_Merit_Cap_Share": [10.0, 0.0, 5.0, 0.0],
            "Career_Merit_Cap_Share": [10.0, 0.0, 5.0, 0.0],
            "Peak_Overall": [70.0, 60.0, 68.0, 58.0],
            "Career_Games_Played": [120.0, 20.0, 96.0, 8.0],
            "Economic_Success": [1, 0, 1, 0],
        }
    )

    metrics = evaluate_complete_model(
        y_true=y_true,
        y_pred=y_pred,
        draft_year=draft_group,
        outcome_columns=outcomes,
    )

    assert "complete_mean_ndcg_at_32" in metrics
    assert "complete_top64_actual_value" in metrics
    assert "complete_top64_weighted_mae_normalized" in metrics
    assert "complete_precision_at_32_positive_value" in metrics
    assert "complete_draft_value_score" in metrics
    assert metrics["complete_bust_rate_at_32"] == pytest.approx(0.5)
    assert "complete_econ_mean_ndcg_at_64" in metrics
    assert "complete_talent_mean_ndcg_at_64" in metrics
    assert "complete_longevity_mean_ndcg_at_64" in metrics


def test_evaluate_complete_model_renames_elite_metrics_with_complete_prefix():
    y_true = np.array([12.0, 6.0, 2.0, 0.0])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1])
    draft_group = np.array(["A:2020", "A:2020", "B:2021", "B:2021"], dtype=object)
    outcomes = pl.DataFrame(
        {
            "Career_Merit_Cap_Share": [12.0, 6.0, 2.0, 0.0],
            "Economic_Success": [1, 1, 1, 0],
        }
    )
    meta = pl.DataFrame({"Position_Group": ["QB", "QB", "WR", "WR"]})
    elite_cfg = {
        "enabled": True,
        "source_column": "Career_Merit_Cap_Share",
        "quantile": 0.75,
        "scope": "position_group",
        "scope_column": "Position_Group",
        "fallback_scope": "global",
        "min_group_size": 2,
        "top_k_precision": 32,
        "top_k_recall": 64,
    }

    metrics = evaluate_complete_model(
        y_true=y_true,
        y_pred=y_pred,
        draft_year=draft_group,
        outcome_columns=outcomes,
        meta_columns=meta,
        elite_cfg=elite_cfg,
    )

    assert "complete_elite_precision_at_32" in metrics
    assert "complete_elite_recall_at_64" in metrics


def test_evaluate_complete_model_by_slice_returns_position_rows():
    y_true = np.array([10.0, 0.0, 6.0, 3.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    draft_group = np.array(["A:2020", "A:2020", "B:2021", "B:2021"], dtype=object)
    slice_values = np.array(["QB", "QB", "WR", "WR"], dtype=object)
    outcomes = pl.DataFrame(
        {
            "Positive_Career_Merit_Cap_Share": [10.0, 0.0, 6.0, 3.0],
            "Career_Merit_Cap_Share": [10.0, 0.0, 6.0, 3.0],
            "Peak_Overall": [70.0, 60.0, 69.0, 61.0],
            "Career_Games_Played": [120.0, 12.0, 96.0, 50.0],
            "Economic_Success": [1, 0, 1, 1],
        }
    )
    meta = pl.DataFrame({"Position_Group": ["QB", "QB", "WR", "WR"]})

    metrics = evaluate_complete_model_by_slice(
        y_true=y_true,
        y_pred=y_pred,
        draft_year=draft_group,
        slice_values=slice_values,
        outcome_columns=outcomes,
        meta_columns=meta,
    )

    assert set(metrics.get_column("slice_value").to_list()) == {"QB", "WR"}
    assert "complete_mean_ndcg_at_64" in metrics.columns
    assert "complete_bust_rate_at_32" in metrics.columns


def test_load_role_bundle_prefers_local_bundle_for_regressor(monkeypatch, tmp_path):
    run_id = "reg-local"
    bundle_dir = tmp_path / "outputs" / "local_model_bundles" / run_id / "regressor"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "model.cbm").write_text("placeholder", encoding="utf-8")
    (bundle_dir / "bundle_metadata.json").write_text(
        '{"model_format":"catboost_native","model_path":"model.cbm"}',
        encoding="utf-8",
    )
    (bundle_dir / "feature_schema.json").write_text(
        '{"feature_columns":["a"],"categorical_columns":[],"excluded_metadata_columns":[],"excluded_target_columns":[]}',
        encoding="utf-8",
    )

    class _FakeCatBoostRegressor:
        def __init__(self):
            self.loaded_path = None

        def load_model(self, path):
            self.loaded_path = path

    monkeypatch.setattr(
        "fof8_ml.evaluation.complete_model.cb.CatBoostRegressor",
        _FakeCatBoostRegressor,
    )
    monkeypatch.setattr(
        "fof8_ml.evaluation.complete_model._require_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not download")),
    )

    client = type(
        "Client",
        (),
        {
            "get_run": lambda self, _run_id: type(
                "Run",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "tags": {"model_role": "regressor"},
                            "params": {
                                "model.name": "catboost_regressor_rmse",
                                "target.regressor.target_space": "raw",
                            },
                        },
                    )()
                },
            )()
        },
    )()

    bundle = _load_role_bundle(
        client,
        run_id,
        "regressor",
        "regressor_model",
        exp_root=str(tmp_path),
    )

    assert isinstance(bundle, LoadedRegressorBundle)
    assert bundle.run_id == run_id
    assert bundle.target_space == "raw"
