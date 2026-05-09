from collections.abc import Iterator
from typing import Any, cast

import mlflow
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from fof8_ml.evaluation.metrics import calculate_career_threshold_metrics
from fof8_ml.models.base import ModelWrapper
from fof8_ml.models.factory import (
    apply_interactive_progress_params,
    apply_quiet_params,
    get_model_wrapper,
)
from fof8_ml.orchestration.pipeline_types import CVResult


def _resolve_group_key(meta: pl.DataFrame, group_key: str | None) -> np.ndarray | None:
    if not group_key:
        return None
    normalized = group_key.strip().lower()
    if normalized == "universe":
        return meta.get_column("Universe").cast(pl.String).to_numpy()
    if normalized in {"draft_class", "universe_year"}:
        return (
            meta.get_column("Universe").cast(pl.String)
            + ":"
            + meta.get_column("Year").cast(pl.String)
        ).to_numpy()
    if group_key in meta.columns:
        return meta.get_column(group_key).to_numpy()
    raise ValueError(f"Unsupported CV group key '{group_key}'.")


def _classifier_splits(
    X: pl.DataFrame,
    y: np.ndarray,
    cv_cfg: DictConfig,
    seed: int,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, Iterator[tuple[np.ndarray, np.ndarray]]]:
    indices = np.arange(len(X))
    if groups is None:
        splitter = StratifiedKFold(
            n_splits=cv_cfg.n_folds,
            shuffle=cv_cfg.shuffle,
            random_state=seed,
        )
        return indices, splitter.split(indices, y)

    splitter = StratifiedGroupKFold(
        n_splits=cv_cfg.n_folds,
        shuffle=cv_cfg.get("shuffle", True),
        random_state=seed,
    )
    return indices, splitter.split(indices, y, groups)


def _regressor_splits(
    X: pl.DataFrame,
    cv_cfg: DictConfig,
    seed: int,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, Iterator[tuple[np.ndarray, np.ndarray]]]:
    indices = np.arange(len(X))
    if groups is None:
        splitter = KFold(n_splits=cv_cfg.n_folds, shuffle=cv_cfg.shuffle, random_state=seed)
        return indices, splitter.split(indices)

    splitter = GroupKFold(n_splits=cv_cfg.n_folds)
    return indices, splitter.split(indices, groups=groups)


def run_cv_classifier(
    X: pl.DataFrame,
    y: np.ndarray,
    model_cfg: DictConfig,
    cv_cfg: DictConfig,
    seed: int,
    groups: np.ndarray | None = None,
    interactive_progress_every: int = 100,
    use_gpu: bool = False,
    quiet: bool = False,
) -> CVResult:
    """Run classifier CV. When groups are provided, folds respect group boundaries."""
    indices, split_iter = _classifier_splits(X, y, cv_cfg, seed, groups)
    oof_probs = np.zeros(len(X))

    cv_metrics = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if not quiet:
            print(f"--- Classifier Fold {fold} ---")
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        params = cast(dict[str, Any], OmegaConf.to_container(model_cfg.params, resolve=True))
        if quiet:
            params = apply_quiet_params(model_cfg.name, params)
        else:
            params = apply_interactive_progress_params(
                model_cfg.name,
                params,
                catboost_progress_every=interactive_progress_every,
            )

        thread_count = model_cfg.params.get("thread_count", -1)
        model = get_model_wrapper(
            model_cfg.name,
            "classifier",
            seed,
            params,
            use_gpu=use_gpu,
            thread_count=thread_count,
        )

        model.fit(X_cv_train, y_cv_train, X_cv_val, y_cv_val)
        best_iterations.append(model.get_best_iteration())
        y_val_prob = model.predict_proba(X_cv_val)

        oof_probs[val_idx] = y_val_prob

        metrics = calculate_career_threshold_metrics(y_cv_val, y_val_prob)
        cv_metrics.append(metrics)
        for m_name, m_val in metrics.items():
            mlflow.log_metric(f"fold_{fold}_{m_name}", m_val)

    summary_metrics = {}
    for key in cv_metrics[0].keys():
        values = [m[key] for m in cv_metrics]
        summary_metrics[f"mean_{key}"] = float(np.mean(values))
    mlflow.log_metrics(summary_metrics)

    return CVResult(
        oof_predictions=oof_probs,
        best_iterations=best_iterations,
        fold_metrics=cv_metrics,
    )


def run_cv_regressor(
    X: pl.DataFrame,
    y: np.ndarray,
    model_cfg: DictConfig,
    cv_cfg: DictConfig,
    seed: int,
    groups: np.ndarray | None = None,
    interactive_progress_every: int = 100,
    use_gpu: bool = False,
    quiet: bool = False,
    target_space: str = "log",
) -> CVResult:
    """Run regressor CV. When groups are provided, folds respect group boundaries."""
    if target_space not in {"log", "raw"}:
        raise ValueError(f"Unsupported target_space '{target_space}'. Expected 'log' or 'raw'.")

    indices, split_iter = _regressor_splits(X, cv_cfg, seed, groups)
    oof_preds = np.zeros(len(X))

    cv_metrics = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if not quiet:
            print(f"--- Regressor Fold {fold} ---")
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        params = cast(dict[str, Any], OmegaConf.to_container(model_cfg.params, resolve=True))
        if quiet:
            params = apply_quiet_params(model_cfg.name, params)
        else:
            params = apply_interactive_progress_params(
                model_cfg.name,
                params,
                catboost_progress_every=interactive_progress_every,
            )

        thread_count = model_cfg.params.get("thread_count", -1)
        model = get_model_wrapper(
            model_cfg.name,
            "regressor",
            seed,
            params,
            use_gpu=use_gpu,
            thread_count=thread_count,
        )

        model.fit(X_cv_train, y_cv_train, X_cv_val, y_cv_val)
        best_iterations.append(model.get_best_iteration())
        y_val_pred = model.predict(X_cv_val)

        oof_preds[val_idx] = y_val_pred

        if target_space == "log":
            y_val_real = np.expm1(y_cv_val)
            y_val_pred_real = np.expm1(y_val_pred)
        else:
            y_val_real = y_cv_val
            y_val_pred_real = np.maximum(y_val_pred, 0)

        rmse = float(np.sqrt(mean_squared_error(y_val_real, y_val_pred_real)))
        mae = float(mean_absolute_error(y_val_real, y_val_pred_real))

        cv_metrics.append({"rmse": rmse, "mae": mae})

        mlflow.log_metric(f"fold_{fold}_rmse", rmse)
        mlflow.log_metric(f"fold_{fold}_mae", mae)

    return CVResult(
        oof_predictions=oof_preds,
        best_iterations=best_iterations,
        fold_metrics=cv_metrics,
    )


def train_final_model(
    model_cfg: DictConfig,
    role: str,
    X: pl.DataFrame,
    y: np.ndarray,
    avg_best_iterations: int,
    seed: int,
    interactive_progress_every: int = 100,
    use_gpu: bool = False,
    quiet: bool = False,
) -> ModelWrapper:
    """Train the final full-dataset model using averaged best iterations from CV."""
    final_params = cast(dict[str, Any], OmegaConf.to_container(model_cfg.params, resolve=True))
    final_params.pop("early_stopping_rounds", None)

    name_lower = model_cfg.name.lower()
    if "catboost" in name_lower:
        final_params["iterations"] = avg_best_iterations
    elif "xgb" in name_lower:
        final_params["n_estimators"] = avg_best_iterations

    if quiet:
        final_params = apply_quiet_params(model_cfg.name, final_params)
    else:
        final_params = apply_interactive_progress_params(
            model_cfg.name,
            final_params,
            catboost_progress_every=interactive_progress_every,
        )

    model = get_model_wrapper(
        model_cfg.name,
        role,
        seed,
        final_params,
        use_gpu=use_gpu,
        thread_count=model_cfg.params.get("thread_count", -1),
    )
    model.fit(X, y)
    return model


def train_model_with_validation(
    model_cfg: DictConfig,
    role: str,
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_val: pl.DataFrame,
    y_val: np.ndarray,
    seed: int,
    interactive_progress_every: int = 100,
    use_gpu: bool = False,
    quiet: bool = False,
) -> ModelWrapper:
    """Train a single model with an explicit validation set for early stopping."""
    params = cast(dict[str, Any], OmegaConf.to_container(model_cfg.params, resolve=True))
    if quiet:
        params = apply_quiet_params(model_cfg.name, params)
    else:
        params = apply_interactive_progress_params(
            model_cfg.name,
            params,
            catboost_progress_every=interactive_progress_every,
        )

    model = get_model_wrapper(
        model_cfg.name,
        role,
        seed,
        params,
        use_gpu=use_gpu,
        thread_count=model_cfg.params.get("thread_count", -1),
    )
    model.fit(X_train, y_train, X_val, y_val)
    return model
