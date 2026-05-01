import mlflow
import numpy as np
import polars as pl

from fof8_ml.orchestration.evaluator import compute_stage2_oof_metrics
from fof8_ml.orchestration.pipeline_runner import PipelineContext
from fof8_ml.orchestration.trainer import run_cv_regressor, train_final_model


def run_regressor_stage(ctx: PipelineContext) -> dict[str, float]:
    cfg = ctx.cfg
    data = ctx.data
    quiet = ctx.sweep_context.quiet

    s1_mask = (data.y_cls == 1).astype(bool)
    X_reg = data.X_train.filter(pl.Series(s1_mask))
    y_reg_target = np.log1p(data.y_reg[s1_mask])

    if not quiet:
        n_hits = int(s1_mask.sum())
        print(f"\nFiltered to {n_hits} ground truth positive cases for regressor training.")
        print("\n" + "=" * 40)
        print("STAGE 2: INTENSITY REGRESSOR")
        print("=" * 40)

    with ctx.logger.start_stage_run("stage2", ctx.sweep_context):
        ctx.logger.log_stage_params(cfg.model, prefix="s2")

        s2_cv_result = run_cv_regressor(
            X=X_reg,
            y=y_reg_target,
            model_cfg=cfg.model,
            cv_cfg=cfg.cv,
            seed=cfg.seed,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
        )

        s2_metrics = compute_stage2_oof_metrics(
            y_true_log=y_reg_target,
            oof_predictions_log=s2_cv_result.oof_predictions,
        )

        cv_rmse = [m["rmse"] for m in s2_cv_result.fold_metrics]
        cv_mae = [m["mae"] for m in s2_cv_result.fold_metrics]
        s2_metrics["s2_mean_rmse"] = float(np.mean(cv_rmse))
        s2_metrics["s2_mean_mae"] = float(np.mean(cv_mae))

        avg_best_iters = (
            int(np.mean(s2_cv_result.best_iterations)) if s2_cv_result.best_iterations else 100
        )
        s2_model = train_final_model(
            model_cfg=cfg.model,
            stage="stage2",
            X=X_reg,
            y=y_reg_target,
            avg_best_iterations=avg_best_iters,
            seed=cfg.seed,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
        )

        ctx.logger.log_regressor_results(s2_metrics, s2_model, X_reg, cfg, quiet)

        mlflow.log_metrics(
            {
                "s2_oof_rmse": s2_metrics["s2_oof_rmse"],
                "s2_oof_mae": s2_metrics["s2_oof_mae"],
            }
        )

    return {
        "s2_oof_rmse": s2_metrics["s2_oof_rmse"],
        "s2_oof_mae": s2_metrics["s2_oof_mae"],
        "s2_mean_rmse": s2_metrics["s2_mean_rmse"],
        "s2_mean_mae": s2_metrics["s2_mean_mae"],
    }
