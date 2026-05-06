import mlflow
import numpy as np
import polars as pl

from fof8_ml.models.registry import get_model_family
from fof8_ml.orchestration.evaluator import compute_cross_outcome_metrics, compute_regressor_oof_metrics
from fof8_ml.orchestration.pipeline_runner import PipelineContext
from fof8_ml.orchestration.trainer import run_cv_regressor, train_final_model


def run_regressor(ctx: PipelineContext) -> dict[str, float]:
    cfg = ctx.cfg
    data = ctx.data
    quiet = ctx.sweep_context.quiet

    positive_mask = (data.y_cls == 1).astype(bool)
    X_reg = data.X_train.filter(pl.Series(positive_mask))
    if "Year" not in data.meta_train.columns:
        raise ValueError("PreparedData.meta_train must include 'Year' for draft-aware metrics.")
    draft_year = data.meta_train["Year"].to_numpy()[positive_mask]
    regressor_cfg = cfg.target.regressor_intensity
    target_space_value = (
        regressor_cfg.get("target_space", "log")
        if hasattr(regressor_cfg, "get")
        else getattr(regressor_cfg, "target_space", "log")
    )
    target_space = str(target_space_value).strip().lower()
    if target_space not in {"raw", "log"}:
        raise ValueError(
            f"Unsupported regressor target_space '{target_space}'. Expected 'raw' or 'log'."
        )

    model_family = get_model_family(role="regressor", model_name=cfg.model.name)
    loss_function = str(getattr(cfg.model.params, "loss_function", ""))
    if model_family == "catboost" and loss_function.startswith("Tweedie") and target_space != "raw":
        raise ValueError(
            "CatBoost Tweedie requires target.regressor_intensity.target_space='raw'. "
            f"Got '{target_space}'."
        )

    y_reg_raw = data.y_reg[positive_mask]
    if model_family == "catboost" and loss_function.startswith("Tweedie") and np.any(y_reg_raw < 0):
        raise ValueError(
            "CatBoost Tweedie requires non-negative regressor targets, but negative values were "
            "found in the configured regressor target column."
        )
    y_reg_target = y_reg_raw if target_space == "raw" else np.log1p(y_reg_raw)

    if not quiet:
        n_hits = int(positive_mask.sum())
        print(f"\nFiltered to {n_hits} ground truth positive cases for regressor training.")
        print(f"Regressor target space: {target_space}")
        print("\n" + "=" * 40)
        print("REGRESSOR: INTENSITY REGRESSOR")
        print("=" * 40)

    with ctx.logger.start_model_run("regressor", ctx.sweep_context):
        ctx.logger.log_model_params(cfg.model, prefix="regressor")
        mlflow.log_param("target.regressor.target_col", cfg.target.regressor_intensity.target_col)
        mlflow.log_param("target.regressor.target_space", target_space)
        mlflow.log_param("regressor_target_space", target_space)

        regressor_cv_result = run_cv_regressor(
            X=X_reg,
            y=y_reg_target,
            model_cfg=cfg.model,
            cv_cfg=cfg.cv,
            seed=cfg.seed,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
            target_space=target_space,
        )

        regressor_metrics = compute_regressor_oof_metrics(
            y_true=y_reg_target,
            oof_predictions=regressor_cv_result.oof_predictions,
            target_space=target_space,
            draft_year=draft_year,
        )
        y_score_raw = (
            np.expm1(regressor_cv_result.oof_predictions)
            if target_space == "log"
            else np.maximum(regressor_cv_result.oof_predictions, 0)
        )
        outcomes_positive = (
            data.outcomes_train.filter(pl.Series(positive_mask))
            if data.outcomes_train is not None
            else None
        )
        regressor_metrics.update(
            compute_cross_outcome_metrics(
                y_score=y_score_raw,
                outcome_columns=outcomes_positive,
                draft_year=draft_year,
            )
        )

        cv_rmse = [m["rmse"] for m in regressor_cv_result.fold_metrics]
        cv_mae = [m["mae"] for m in regressor_cv_result.fold_metrics]
        regressor_metrics["regressor_mean_rmse"] = float(np.mean(cv_rmse))
        regressor_metrics["regressor_mean_mae"] = float(np.mean(cv_mae))

        avg_best_iters = (
            int(np.mean(regressor_cv_result.best_iterations))
            if regressor_cv_result.best_iterations
            else 100
        )
        regressor_model = train_final_model(
            model_cfg=cfg.model,
            role="regressor",
            X=X_reg,
            y=y_reg_target,
            avg_best_iterations=avg_best_iters,
            seed=cfg.seed,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
        )

        ctx.logger.log_regressor_results(regressor_metrics, regressor_model, X_reg, cfg, quiet)

        mlflow.log_metrics(
            {
                "regressor_oof_rmse": regressor_metrics["regressor_oof_rmse"],
                "regressor_oof_mae": regressor_metrics["regressor_oof_mae"],
                "regressor_draft_value_score": regressor_metrics["regressor_draft_value_score"],
            }
        )

    return {
        "regressor_oof_rmse": regressor_metrics["regressor_oof_rmse"],
        "regressor_oof_mae": regressor_metrics["regressor_oof_mae"],
        "regressor_mean_rmse": regressor_metrics["regressor_mean_rmse"],
        "regressor_mean_mae": regressor_metrics["regressor_mean_mae"],
        "regressor_draft_value_score": regressor_metrics["regressor_draft_value_score"],
    }
