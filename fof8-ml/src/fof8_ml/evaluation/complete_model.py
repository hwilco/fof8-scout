from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from fof8_ml.data.preprocessing import ScalerLike, preprocess_for_sklearn
from fof8_ml.data.schema import FEATURE_SCHEMA_ARTIFACT_PATH, FeatureSchema
from fof8_ml.evaluation.metrics import (
    calibration_slope,
    mean_ndcg_by_group,
    mean_topk_bias_by_group,
    mean_topk_weighted_mae_normalized_by_group,
)
from fof8_ml.models.base import (
    ModelFamily,
    ModelRole,
    SupportedClassifierModel,
    SupportedRegressorModel,
    SupportedSklearnRegressorModel,
)
from fof8_ml.models.calibration import BetaCalibrator
from fof8_ml.models.registry import get_model_family
from fof8_ml.orchestration.evaluator import compute_cross_outcome_metrics


@dataclass
class LoadedClassifierBundle:
    role: ModelRole
    run_id: str
    family: ModelFamily
    model: SupportedClassifierModel
    schema: FeatureSchema
    threshold: float
    calibrator: BetaCalibrator


@dataclass
class LoadedRegressorBundle:
    role: ModelRole
    run_id: str
    family: ModelFamily
    model: SupportedRegressorModel
    schema: FeatureSchema
    target_space: str
    sklearn_scaler: ScalerLike | None = None
    sklearn_expected_columns: list[str] | None = None


@dataclass
class CompleteModelBundle:
    classifier: LoadedClassifierBundle
    regressor: LoadedRegressorBundle


def _require_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> str:
    try:
        return client.download_artifacts(run_id, artifact_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Run '{run_id}' is missing required artifact '{artifact_path}'."
        ) from exc


def _resolve_model_uri(client: MlflowClient, run_id: str, artifact_path: str) -> str:
    runs_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        artifacts = client.list_artifacts(run_id, artifact_path)
        if any(artifact.path == f"{artifact_path}/MLmodel" for artifact in artifacts):
            return runs_uri
    except Exception:
        pass

    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id
    try:
        logged_models = client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
            max_results=100,
        )
    except Exception:
        logged_models = client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"name = '{artifact_path}'",
            max_results=100,
        )

    matches = [
        model
        for model in logged_models
        if model.name == artifact_path and model.source_run_id == run_id
    ]
    if not matches:
        return runs_uri

    best_model = max(matches, key=lambda model: model.last_updated_timestamp or 0)
    return f"models:/{best_model.model_id}"


def _load_model_by_family(
    model_uri: str, family: ModelFamily
) -> SupportedClassifierModel | SupportedRegressorModel:
    if family == "catboost":
        model = mlflow.catboost.load_model(model_uri)
        assert model is not None
        return cast(SupportedClassifierModel | SupportedRegressorModel, model)
    if family == "xgb":
        model = mlflow.xgboost.load_model(model_uri)
        assert model is not None
        if not hasattr(model, "feature_names_in_") and hasattr(model, "feature_names"):
            setattr(model, "feature_names_in_", np.array(model.feature_names))
        return cast(SupportedClassifierModel | SupportedRegressorModel, model)
    if family == "sklearn":
        model = mlflow.sklearn.load_model(model_uri)
        assert model is not None
        return cast(SupportedSklearnRegressorModel, model)
    raise ValueError(f"Unsupported model family '{family}'.")


def load_feature_schema(client: MlflowClient, run_id: str) -> FeatureSchema:
    schema_local_path = _require_artifact(client, run_id, FEATURE_SCHEMA_ARTIFACT_PATH)
    with open(schema_local_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return FeatureSchema.from_dict(payload)


def _load_role_bundle(
    client: MlflowClient,
    run_id: str,
    role: ModelRole,
    artifact_path: str,
) -> LoadedClassifierBundle | LoadedRegressorBundle:
    run = client.get_run(run_id)
    actual_role = run.data.tags.get("model_role")
    if actual_role != role:
        raise ValueError(f"Run '{run_id}' has model_role={actual_role!r}, expected '{role}'.")

    model_name = str(run.data.params.get("model.name", ""))
    family = get_model_family(role=role, model_name=model_name)
    if family is None:
        raise ValueError(
            f"Run '{run_id}' uses unknown {role} model '{model_name}'. "
            "Update fof8_ml.models.registry before evaluating this run."
        )

    model_uri = _resolve_model_uri(client, run_id, artifact_path)
    model = _load_model_by_family(model_uri, family)
    schema = load_feature_schema(client, run_id)

    if role == "classifier":
        calibrator = cast(
            BetaCalibrator,
            joblib.load(_require_artifact(client, run_id, "classifier_beta_calibrator.joblib")),
        )
        return LoadedClassifierBundle(
            role=role,
            run_id=run_id,
            family=family,
            model=cast(SupportedClassifierModel, model),
            schema=schema,
            threshold=float(run.data.params.get("classifier_optimal_threshold", 0.5)),
            calibrator=calibrator,
        )
    sklearn_scaler: ScalerLike | None = None
    sklearn_expected_columns: list[str] | None = None
    if family == "sklearn":
        scaler_path = _require_artifact(client, run_id, "regressor_model_scaler.joblib")
        features_path = _require_artifact(client, run_id, "regressor_model_features.txt")
        sklearn_scaler = cast(ScalerLike, joblib.load(scaler_path))
        sklearn_expected_columns = Path(features_path).read_text(encoding="utf-8").splitlines()
    return LoadedRegressorBundle(
        role=role,
        run_id=run_id,
        family=family,
        model=cast(SupportedRegressorModel, model),
        schema=schema,
        target_space=str(run.data.params.get("target.regressor.target_space", "log")),
        sklearn_scaler=sklearn_scaler,
        sklearn_expected_columns=sklearn_expected_columns,
    )


def load_complete_model(
    client: MlflowClient,
    classifier_run_id: str,
    regressor_run_id: str,
) -> CompleteModelBundle:
    return CompleteModelBundle(
        classifier=cast(
            LoadedClassifierBundle,
            _load_role_bundle(client, classifier_run_id, "classifier", "classifier_model"),
        ),
        regressor=cast(
            LoadedRegressorBundle,
            _load_role_bundle(client, regressor_run_id, "regressor", "regressor_model"),
        ),
    )


def load_catboost_complete_model(
    client: MlflowClient,
    classifier_run_id: str,
    regressor_run_id: str,
) -> CompleteModelBundle:
    return load_complete_model(client, classifier_run_id, regressor_run_id)


def _predict_classifier(bundle: LoadedClassifierBundle, X_full: pl.DataFrame) -> np.ndarray:
    features = bundle.schema.apply(X_full)
    raw_probs = bundle.model.predict_proba(features.to_pandas())[:, 1]
    return bundle.calibrator.predict(raw_probs)


def _predict_regressor(bundle: LoadedRegressorBundle, X_full: pl.DataFrame) -> np.ndarray:
    features = bundle.schema.apply(X_full)

    if bundle.family == "sklearn":
        transformed, _, _ = preprocess_for_sklearn(
            features,
            scaler=bundle.sklearn_scaler,
            expected_columns=bundle.sklearn_expected_columns,
        )
        raw_predictions = bundle.model.predict(transformed.to_numpy())
        return np.maximum(raw_predictions.astype(float), 0.0)

    if bundle.family == "xgb":
        raw_predictions = bundle.model.predict(features)
    else:
        raw_predictions = bundle.model.predict(features.to_pandas())

    target_space = (bundle.target_space or "log").strip().lower()
    if target_space == "log":
        return np.expm1(np.asarray(raw_predictions, dtype=float))
    if target_space == "raw":
        return np.maximum(np.asarray(raw_predictions, dtype=float), 0.0)
    raise ValueError(f"Unsupported regressor target_space '{target_space}'.")


def predict_complete_model(
    X_full: pl.DataFrame,
    classifier_bundle: LoadedClassifierBundle,
    regressor_bundle: LoadedRegressorBundle,
) -> dict[str, np.ndarray]:
    classifier_probability = _predict_classifier(classifier_bundle, X_full)
    regressor_prediction = _predict_regressor(regressor_bundle, X_full)
    complete_prediction = classifier_probability * np.maximum(regressor_prediction, 0.0)
    return {
        "classifier_probability": classifier_probability,
        "regressor_prediction": regressor_prediction,
        "complete_prediction": complete_prediction,
    }


def _mean_topk_actual_value_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> float:
    values: list[float] = []
    for group in np.unique(groups):
        mask = groups == group
        if not np.any(mask):
            continue
        y_group = y_true[mask]
        k_eff = min(k, y_group.size)
        order = np.argsort(-y_score[mask])[:k_eff]
        values.append(float(np.sum(y_group[order])))
    return float(np.mean(values)) if values else 0.0


def _mean_topk_positive_precision_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> float:
    values: list[float] = []
    positive = (y_true > 0).astype(float)
    for group in np.unique(groups):
        mask = groups == group
        if not np.any(mask):
            continue
        y_group = positive[mask]
        k_eff = min(k, y_group.size)
        order = np.argsort(-y_score[mask])[:k_eff]
        values.append(float(np.mean(y_group[order])))
    return float(np.mean(values)) if values else 0.0


def _rename_cross_outcome_metrics(metrics: dict[str, float]) -> dict[str, float]:
    renamed: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("cross_"):
            renamed[f"complete_{key.removeprefix('cross_')}"] = value
        else:
            renamed[key] = value
    return renamed


def evaluate_complete_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    draft_year: np.ndarray,
    *,
    outcome_columns: pl.DataFrame | None = None,
    meta_columns: pl.DataFrame | None = None,
    elite_cfg: dict[str, Any] | None = None,
    elite_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    group_key = np.asarray(draft_year, dtype=object)

    metrics = {
        "complete_mean_ndcg_at_32": mean_ndcg_by_group(y_true_arr, y_pred_arr, group_key, k=32),
        "complete_mean_ndcg_at_64": mean_ndcg_by_group(y_true_arr, y_pred_arr, group_key, k=64),
        "complete_top32_actual_value": _mean_topk_actual_value_by_group(
            y_true_arr, y_pred_arr, group_key, k=32
        ),
        "complete_top64_actual_value": _mean_topk_actual_value_by_group(
            y_true_arr, y_pred_arr, group_key, k=64
        ),
        "complete_top32_weighted_mae_normalized": mean_topk_weighted_mae_normalized_by_group(
            y_true_arr, y_pred_arr, group_key, k=32
        ),
        "complete_top64_weighted_mae_normalized": mean_topk_weighted_mae_normalized_by_group(
            y_true_arr, y_pred_arr, group_key, k=64
        ),
        "complete_top64_bias": mean_topk_bias_by_group(y_true_arr, y_pred_arr, group_key, k=64),
        "complete_top64_calibration_slope": calibration_slope(y_true_arr, y_pred_arr),
        "complete_precision_at_32_positive_value": _mean_topk_positive_precision_by_group(
            y_true_arr, y_pred_arr, group_key, k=32
        ),
    }
    metrics["complete_draft_value_score"] = float(
        metrics["complete_mean_ndcg_at_64"]
        - 0.25 * metrics["complete_top64_weighted_mae_normalized"]
    )

    cross_outcome_metrics = compute_cross_outcome_metrics(
        y_score=y_pred_arr,
        outcome_columns=outcome_columns,
        draft_group=group_key,
        meta_columns=meta_columns,
        elite_cfg=elite_cfg,
        elite_thresholds=elite_thresholds,
    )
    metrics.update(_rename_cross_outcome_metrics(cross_outcome_metrics))
    return metrics
