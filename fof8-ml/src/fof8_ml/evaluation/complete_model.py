from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import catboost as cb
import joblib
import mlflow
import numpy as np
import polars as pl
import xgboost as xgb
from mlflow.entities.model_registry import ModelVersion
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
    """Bundle of classifier artifacts needed for complete-model inference.

    Attributes:
        role: Model role tag.
        run_id: Source MLflow run id.
        family: Model family identifier.
        model: Loaded classifier model object.
        schema: Feature schema used to transform input features.
        threshold: Stored classifier threshold from training.
        calibrator: Beta calibrator fit during classifier training.
    """

    role: ModelRole
    run_id: str
    family: ModelFamily
    model: SupportedClassifierModel
    schema: FeatureSchema
    threshold: float
    calibrator: BetaCalibrator


@dataclass
class LoadedRegressorBundle:
    """Bundle of regressor artifacts needed for complete-model inference.

    Attributes:
        role: Model role tag.
        run_id: Source MLflow run id.
        family: Model family identifier.
        model: Loaded regressor model object.
        schema: Feature schema used to transform input features.
        target_space: Target space used during regressor training.
    sklearn_scaler: Optional sklearn preprocessing scaler.
    sklearn_expected_columns: Optional sklearn feature order.
    sklearn_preprocessor: Optional sklearn MLP preprocessing bundle.
    """

    role: ModelRole
    run_id: str
    family: ModelFamily
    model: SupportedRegressorModel
    schema: FeatureSchema
    target_space: str
    sklearn_scaler: ScalerLike | None = None
    sklearn_expected_columns: list[str] | None = None
    sklearn_preprocessor: dict[str, Any] | None = None


@dataclass
class CompleteModelBundle:
    """Container for classifier and regressor bundles used together.

    Attributes:
        classifier: Loaded classifier bundle.
        regressor: Loaded regressor bundle.
    """

    classifier: LoadedClassifierBundle
    regressor: LoadedRegressorBundle


def _local_bundle_dir(exp_root: str, run_id: str, role: ModelRole) -> Path:
    return Path(exp_root) / "outputs" / "local_model_bundles" / run_id / role


def _require_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> str:
    """Download a required run artifact and fail with a clear error if missing.

    Args:
        client: MLflow tracking client.
        run_id: Source run id.
        artifact_path: Path of artifact within the run.

    Returns:
        Local filesystem path to the downloaded artifact.
    """

    try:
        return client.download_artifacts(run_id, artifact_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Run '{run_id}' is missing required artifact '{artifact_path}'."
        ) from exc


def _resolve_model_uri(client: MlflowClient, run_id: str, artifact_path: str) -> str:
    """Resolve the best model URI for a run, preferring explicit run artifacts.

    Args:
        client: MLflow tracking client.
        run_id: Source run id.
        artifact_path: Expected model artifact directory name.

    Returns:
        A loadable MLflow model URI.
    """

    runs_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        artifacts = client.list_artifacts(run_id, artifact_path)
        if any(artifact.path == f"{artifact_path}/MLmodel" for artifact in artifacts):
            return runs_uri
    except Exception:
        pass

    try:
        versions = client.search_model_versions(
            filter_string=f"run_id = '{run_id}'",
            max_results=100,
        )
    except Exception:
        return runs_uri

    matches = [version for version in versions if getattr(version, "run_id", None) == run_id]
    if not matches:
        return runs_uri

    def _version_sort_key(model_version: ModelVersion) -> tuple[int, int]:
        timestamp = int(getattr(model_version, "last_updated_timestamp", 0) or 0)
        try:
            number = int(getattr(model_version, "version", 0) or 0)
        except (TypeError, ValueError):
            number = 0
        return (timestamp, number)

    best_version = max(matches, key=_version_sort_key)
    model_name = str(getattr(best_version, "name", ""))
    model_version = str(getattr(best_version, "version", ""))
    if not model_name or not model_version:
        return runs_uri
    return f"models:/{model_name}/{model_version}"


def _load_model_by_family(
    model_uri: str, family: ModelFamily
) -> SupportedClassifierModel | SupportedRegressorModel:
    """Load a model from MLflow using the correct model-flavor loader.

    Args:
        model_uri: MLflow model URI.
        family: Model family key.

    Returns:
        Loaded model object.
    """

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
    """Load the feature schema artifact for a run.

    Args:
        client: MLflow tracking client.
        run_id: Source run id.

    Returns:
        Parsed feature schema.
    """

    schema_local_path = _require_artifact(client, run_id, FEATURE_SCHEMA_ARTIFACT_PATH)
    with open(schema_local_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return FeatureSchema.from_dict(payload)


def _load_role_bundle(
    client: MlflowClient,
    run_id: str,
    role: ModelRole,
    artifact_path: str,
    exp_root: str | None = None,
    require_local_bundle: bool = False,
) -> LoadedClassifierBundle | LoadedRegressorBundle:
    """Load all model artifacts needed for a single role.

    Args:
        client: MLflow tracking client.
        run_id: Source run id.
        role: Expected role label.
        artifact_path: Model artifact directory for this role.

    Returns:
        Loaded classifier or regressor bundle.
    """

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

    local_bundle = _local_bundle_dir(exp_root, run_id, role) if exp_root else None
    metadata_path = local_bundle / "bundle_metadata.json" if local_bundle else None
    use_local_bundle = bool(metadata_path and metadata_path.exists())
    if require_local_bundle and not use_local_bundle:
        expected = str(metadata_path) if metadata_path is not None else "<no exp_root provided>"
        raise FileNotFoundError(
            f"Local model bundle is required for {role} run '{run_id}', but "
            f"bundle_metadata.json was not found at {expected}. "
            "Rerun the source model with local bundle export enabled or disable "
            "diagnostics.skip_mlflow_model_logging for this matrix."
        )

    if use_local_bundle:
        assert local_bundle is not None
        assert metadata_path is not None
        with open(metadata_path, encoding="utf-8") as handle:
            bundle_metadata = json.load(handle)
        with open(local_bundle / FEATURE_SCHEMA_ARTIFACT_PATH, encoding="utf-8") as handle:
            schema = FeatureSchema.from_dict(json.load(handle))
        model_format = str(bundle_metadata.get("model_format", ""))
        model_rel_path = str(bundle_metadata.get("model_path", ""))
        model_path = local_bundle / model_rel_path
        if family == "catboost":
            if role == "classifier":
                model = cb.CatBoostClassifier()
            else:
                model = cb.CatBoostRegressor()
            model.load_model(str(model_path))
        elif family == "xgb":
            if role == "classifier":
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
            model.load_model(str(model_path))
        elif model_format == "joblib":
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported local bundle model_format '{model_format}'.")
    else:
        model_uri = _resolve_model_uri(client, run_id, artifact_path)
        model = _load_model_by_family(model_uri, family)
        schema = load_feature_schema(client, run_id)

    if role == "classifier":
        calibrator_path = (
            local_bundle / "classifier_beta_calibrator.joblib"
            if use_local_bundle and local_bundle is not None
            else Path(_require_artifact(client, run_id, "classifier_beta_calibrator.joblib"))
        )
        calibrator = cast(BetaCalibrator, joblib.load(calibrator_path))
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
    sklearn_preprocessor: dict[str, Any] | None = None
    if family == "sklearn":
        if use_local_bundle and local_bundle is not None:
            preprocessor_path = local_bundle / "regressor_model_preprocessor.joblib"
            if preprocessor_path.exists():
                sklearn_preprocessor = cast(dict[str, Any], joblib.load(preprocessor_path))
            else:
                scaler_path = str(local_bundle / "regressor_model_scaler.joblib")
                features_path = str(local_bundle / "regressor_model_features.txt")
                sklearn_scaler = cast(ScalerLike, joblib.load(scaler_path))
                sklearn_expected_columns = (
                    Path(features_path).read_text(encoding="utf-8").splitlines()
                )
        else:
            try:
                preprocessor_path = _require_artifact(
                    client, run_id, "regressor_model_preprocessor.joblib"
                )
                sklearn_preprocessor = cast(dict[str, Any], joblib.load(preprocessor_path))
            except FileNotFoundError:
                scaler_path = _require_artifact(client, run_id, "regressor_model_scaler.joblib")
                features_path = _require_artifact(client, run_id, "regressor_model_features.txt")
                sklearn_scaler = cast(ScalerLike, joblib.load(scaler_path))
                sklearn_expected_columns = (
                    Path(features_path).read_text(encoding="utf-8").splitlines()
                )
    return LoadedRegressorBundle(
        role=role,
        run_id=run_id,
        family=family,
        model=cast(SupportedRegressorModel, model),
        schema=schema,
        target_space=str(run.data.params.get("target.regressor.target_space", "log")),
        sklearn_scaler=sklearn_scaler,
        sklearn_expected_columns=sklearn_expected_columns,
        sklearn_preprocessor=sklearn_preprocessor,
    )


def load_complete_model(
    client: MlflowClient,
    classifier_run_id: str,
    regressor_run_id: str,
    exp_root: str | None = None,
    require_local_bundles: bool = False,
) -> CompleteModelBundle:
    """Load the classifier and regressor bundles for complete-model scoring.

    Args:
        client: MLflow tracking client.
        classifier_run_id: Classifier run id.
        regressor_run_id: Regressor run id.

    Returns:
        Combined complete-model bundle.
    """

    return CompleteModelBundle(
        classifier=cast(
            LoadedClassifierBundle,
            _load_role_bundle(
                client,
                classifier_run_id,
                "classifier",
                "classifier_model",
                exp_root=exp_root,
                require_local_bundle=require_local_bundles,
            ),
        ),
        regressor=cast(
            LoadedRegressorBundle,
            _load_role_bundle(
                client,
                regressor_run_id,
                "regressor",
                "regressor_model",
                exp_root=exp_root,
                require_local_bundle=require_local_bundles,
            ),
        ),
    )


def load_catboost_complete_model(
    client: MlflowClient,
    classifier_run_id: str,
    regressor_run_id: str,
    exp_root: str | None = None,
) -> CompleteModelBundle:
    """Backwards-compatible wrapper for complete-model loading.

    Args:
        client: MLflow tracking client.
        classifier_run_id: Classifier run id.
        regressor_run_id: Regressor run id.

    Returns:
        Combined complete-model bundle.
    """

    return load_complete_model(client, classifier_run_id, regressor_run_id, exp_root=exp_root)


def _predict_classifier(bundle: LoadedClassifierBundle, X_full: pl.DataFrame) -> np.ndarray:
    """Predict calibrated classifier probabilities for full-feature rows.

    Args:
        bundle: Loaded classifier bundle.
        X_full: Full input features.

    Returns:
        Calibrated positive-class probabilities.
    """

    features = bundle.schema.apply(X_full)
    raw_probs = bundle.model.predict_proba(features.to_numpy())[:, 1]
    return bundle.calibrator.predict(raw_probs)


def _preprocess_for_sklearn_mlp(
    X: pl.DataFrame,
    preprocessor: dict[str, Any],
) -> pl.DataFrame:
    """Apply the dense sklearn MLP preprocessor saved by SklearnMLPRegressorWrapper."""

    columns = cast(list[str], preprocessor["columns"])
    categorical_columns = cast(list[str], preprocessor.get("categorical_columns", []))
    numeric_columns = cast(list[str], preprocessor.get("numeric_columns", []))
    missing_indicator_columns = set(
        cast(list[str], preprocessor.get("missing_indicator_columns", []))
    )
    numeric_medians = cast(dict[str, float], preprocessor.get("numeric_medians", {}))
    scaler = cast(ScalerLike, preprocessor["scaler"])

    numeric_exprs = [
        pl.col(col).cast(pl.Float64).fill_null(float(numeric_medians.get(col, 0.0))).alias(col)
        for col in numeric_columns
        if col in X.columns
    ]
    indicator_exprs = [
        pl.col(col).is_null().cast(pl.Int8).alias(f"{col}__missing")
        for col in numeric_columns
        if f"{col}__missing" in missing_indicator_columns and col in X.columns
    ]

    frames: list[pl.DataFrame] = []
    if numeric_exprs:
        frames.append(X.select(numeric_exprs))
    if indicator_exprs:
        frames.append(X.select(indicator_exprs))
    existing_categoricals = [col for col in categorical_columns if col in X.columns]
    if existing_categoricals:
        frames.append(X.select(existing_categoricals).to_dummies(drop_first=False))

    X_prepared = pl.concat(frames, how="horizontal") if frames else pl.DataFrame()
    missing_cols = [col for col in columns if col not in X_prepared.columns]
    if missing_cols:
        X_prepared = X_prepared.with_columns([pl.lit(0).alias(col) for col in missing_cols])
    X_prepared = X_prepared.select(columns)
    return pl.DataFrame(np.asarray(scaler.transform(X_prepared.to_numpy())), schema=columns)


def _predict_regressor(bundle: LoadedRegressorBundle, X_full: pl.DataFrame) -> np.ndarray:
    """Predict non-negative regression values in raw output space.

    Args:
        bundle: Loaded regressor bundle.
        X_full: Full input features.

    Returns:
        Regressor predictions converted into raw target space.
    """

    features = bundle.schema.apply(X_full)

    if bundle.family == "sklearn":
        if bundle.sklearn_preprocessor is not None:
            transformed = _preprocess_for_sklearn_mlp(features, bundle.sklearn_preprocessor)
        else:
            transformed, _, _ = preprocess_for_sklearn(
                features,
                scaler=bundle.sklearn_scaler,
                expected_columns=bundle.sklearn_expected_columns,
            )
        raw_predictions = bundle.model.predict(transformed.to_numpy())
        return np.maximum(np.asarray(raw_predictions, dtype=float), 0.0)

    if bundle.family == "xgb":
        raw_predictions = bundle.model.predict(features.to_numpy())
    else:
        raw_predictions = bundle.model.predict(features.to_numpy())

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
    """Run classifier and regressor predictions and combine into complete score.

    Args:
        X_full: Full input features.
        classifier_bundle: Loaded classifier bundle.
        regressor_bundle: Loaded regressor bundle.

    Returns:
        Dictionary with classifier, regressor, and combined predictions.
    """

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
    """Compute mean top-k realized value across groups.

    Args:
        y_true: True values.
        y_score: Ranking scores.
        groups: Group labels for per-group ranking.
        k: Top-k cutoff.

    Returns:
        Mean top-k summed true value across groups.
    """

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
    """Compute mean top-k positive-rate precision across groups.

    Args:
        y_true: True values.
        y_score: Ranking scores.
        groups: Group labels for per-group ranking.
        k: Top-k cutoff.

    Returns:
        Mean top-k positive precision across groups.
    """

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
    """Rename cross-outcome metrics into complete-model metric namespace.

    Args:
        metrics: Raw cross-outcome metrics.

    Returns:
        Metrics dictionary with complete-model key prefixes.
    """

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
    """Compute complete-model evaluation metrics on a held-out split.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted complete-model values.
        draft_year: Group key for within-group ranking metrics.
        outcome_columns: Optional outcome columns used for cross-outcome metrics.
        meta_columns: Optional metadata columns used for scoped metrics.
        elite_cfg: Optional elite-metric config.
        elite_thresholds: Optional pre-fit elite thresholds.

    Returns:
        Dictionary of aggregate complete-model metrics.
    """

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


def evaluate_complete_model_by_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    draft_year: np.ndarray,
    slice_values: np.ndarray,
    *,
    slice_column: str = "Position_Group",
    outcome_columns: pl.DataFrame | None = None,
    meta_columns: pl.DataFrame | None = None,
    elite_cfg: dict[str, Any] | None = None,
    elite_thresholds: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Compute complete-model scorecards for each slice value.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted complete-model values.
        draft_year: Group key for within-group ranking metrics.
        slice_values: Slice label for each row, usually `Position_Group`.
        slice_column: Name of the slice column for output labeling.
        outcome_columns: Optional outcome columns used for cross-outcome metrics.
        meta_columns: Optional metadata columns used for scoped elite metrics.
        elite_cfg: Optional elite-metric config.
        elite_thresholds: Optional pre-fit elite thresholds.

    Returns:
        A Polars dataframe containing one metric row per slice value.
    """

    rows: list[dict[str, Any]] = []
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    draft_group_arr = np.asarray(draft_year, dtype=object)
    slice_arr = np.asarray(slice_values, dtype=object)

    for raw_slice_value in np.unique(slice_arr):
        mask = slice_arr == raw_slice_value
        if not np.any(mask):
            continue

        slice_outcomes = (
            outcome_columns.filter(pl.Series(mask)) if outcome_columns is not None else None
        )
        slice_meta = meta_columns.filter(pl.Series(mask)) if meta_columns is not None else None
        metrics = evaluate_complete_model(
            y_true=y_true_arr[mask],
            y_pred=y_pred_arr[mask],
            draft_year=draft_group_arr[mask],
            outcome_columns=slice_outcomes,
            meta_columns=slice_meta,
            elite_cfg=elite_cfg,
            elite_thresholds=elite_thresholds,
        )
        row: dict[str, Any] = {
            "slice_column": slice_column,
            "slice_value": str(raw_slice_value),
            "n_players": int(np.sum(mask)),
            "n_draft_classes": int(np.unique(draft_group_arr[mask]).size),
        }
        row.update(metrics)
        rows.append(row)

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            {
                "slice_column": [],
                "slice_value": [],
                "n_players": [],
                "n_draft_classes": [],
            }
        )
    )
