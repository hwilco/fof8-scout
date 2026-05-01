import mlflow
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from fof8_ml.evaluation.metrics import calculate_career_threshold_metrics
from fof8_ml.models.base import ModelWrapper
from fof8_ml.models.factory import apply_quiet_params, get_model_wrapper
from fof8_ml.orchestration.pipeline_types import CVResult


def run_cv_classifier(
    X: pl.DataFrame,
    y: np.ndarray,
    model_cfg: DictConfig,
    cv_cfg: DictConfig,
    seed: int,
    use_gpu: bool = False,
    quiet: bool = False,
) -> CVResult:
    """Run stratified k-fold CV for a classifier. Returns CVResult with OOF probabilities."""
    skf = StratifiedKFold(n_splits=cv_cfg.n_folds, shuffle=cv_cfg.shuffle, random_state=seed)
    indices = np.arange(len(X))
    oof_probs = np.zeros(len(X))

    cv_metrics = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y)):
        if not quiet:
            print(f"--- S1 Fold {fold} ---")
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        params = OmegaConf.to_container(model_cfg.params, resolve=True)
        if quiet:
            params = apply_quiet_params(model_cfg.name, params)

        thread_count = model_cfg.params.get("thread_count", -1)
        model = get_model_wrapper(
            model_cfg.name,
            "stage1",
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
    use_gpu: bool = False,
    quiet: bool = False,
) -> CVResult:
    """Run k-fold CV for a regressor. Returns CVResult with OOF predictions."""
    kf = KFold(n_splits=cv_cfg.n_folds, shuffle=cv_cfg.shuffle, random_state=seed)
    indices = np.arange(len(X))
    oof_preds = np.zeros(len(X))

    cv_metrics = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if not quiet:
            print(f"--- S2 Fold {fold} ---")
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        params = OmegaConf.to_container(model_cfg.params, resolve=True)
        if quiet:
            params = apply_quiet_params(model_cfg.name, params)

        thread_count = model_cfg.params.get("thread_count", -1)
        model = get_model_wrapper(
            model_cfg.name,
            "stage2",
            seed,
            params,
            use_gpu=use_gpu,
            thread_count=thread_count,
        )

        model.fit(X_cv_train, y_cv_train, X_cv_val, y_cv_val)
        best_iterations.append(model.get_best_iteration())
        y_val_pred = model.predict(X_cv_val)

        oof_preds[val_idx] = y_val_pred

        y_val_real = np.expm1(y_cv_val)
        y_val_pred_real = np.expm1(y_val_pred)

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
    stage: str,
    X: pl.DataFrame,
    y: np.ndarray,
    avg_best_iterations: int,
    seed: int,
    use_gpu: bool = False,
    quiet: bool = False,
) -> ModelWrapper:
    """Train the final full-dataset model using averaged best iterations from CV."""
    final_params = OmegaConf.to_container(model_cfg.params, resolve=True)
    final_params.pop("early_stopping_rounds", None)

    name_lower = model_cfg.name.lower()
    if "catboost" in name_lower:
        final_params["iterations"] = avg_best_iterations
    elif "xgb" in name_lower:
        final_params["n_estimators"] = avg_best_iterations

    if quiet:
        final_params = apply_quiet_params(model_cfg.name, final_params)

    model = get_model_wrapper(
        model_cfg.name,
        stage,
        seed,
        final_params,
        use_gpu=use_gpu,
        thread_count=model_cfg.params.get("thread_count", -1),
    )
    model.fit(X, y)
    return model
