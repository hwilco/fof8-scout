import logging
import os
import warnings

import hydra
import matplotlib
import mlflow
import numpy as np
import polars as pl
from fof8_ml.orchestration import (
    DataLoader,
    ExperimentLogger,
    SweepManager,
    compute_stage2_oof_metrics,
    run_cv_regressor,
    train_final_model,
)
from omegaconf import DictConfig

# Suppress Optuna deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
# Suppress Hydra experimental feature warnings
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


def _resolve_exp_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, ".."))


@hydra.main(version_base=None, config_path="conf", config_name="regressor_pipeline")
def main(cfg: DictConfig) -> float:
    """Train and evaluate the Stage 2 Intensity Regressor.

    Training is performed on ground truth positive cases (y_cls == 1) only.

    Args:
        cfg: Hydra config from regressor_pipeline.yaml.

    Returns:
        The optimization metric score for the current trial.
    """
    exp_root = _resolve_exp_root()
    absolute_raw_path = os.path.abspath(os.path.join(exp_root, cfg.data.raw_path))

    # 1. Infrastructure
    logger = ExperimentLogger(exp_root, cfg.experiment_name)
    logger.init_tracking()
    sweep_mgr = SweepManager(logger.client, logger.experiment_id, exp_root)
    ctx = sweep_mgr.detect_sweep(cfg)

    # 2. Data
    loader = DataLoader(exp_root, quiet=ctx.quiet)
    data = loader.load(cfg)
    loader.print_summary(data, cfg)

    # Trial-specific feature ablation
    data = loader.apply_feature_ablation(
        data, cfg.get("include_features"), cfg.get("exclude_features")
    )

    # Filter to ground truth positive cases (survivors) for regressor training
    s1_mask = (data.y_cls == 1).astype(bool)
    X_reg = data.X_train.filter(pl.Series(s1_mask))
    y_reg_target = np.log1p(data.y_reg[s1_mask])

    if not ctx.quiet:
        n_hits = int(s1_mask.sum())
        print(f"\nFiltered to {n_hits} ground truth survivors for regressor training.")

    # 3. Train & Evaluate
    with logger.start_pipeline_run(f"Regressor_{cfg.model.name}", tags=ctx.tags) as pipeline_run:
        logger.log_data_summary(data, cfg, absolute_raw_path, ctx.is_sweep, ctx.trial_num)

        if not ctx.quiet:
            print("\n" + "=" * 40)
            print("STAGE 2: INTENSITY REGRESSOR")
            print("=" * 40)

        with logger.start_stage_run("stage2", ctx):
            logger.log_stage_params(cfg.model, prefix="s2")

            # Cross-validation
            s2_cv_result = run_cv_regressor(
                X=X_reg,
                y=y_reg_target,
                model_cfg=cfg.model,
                cv_cfg=cfg.cv,
                seed=cfg.seed,
                use_gpu=cfg.use_gpu,
                quiet=ctx.quiet,
            )

            # OOF Metrics
            s2_metrics = compute_stage2_oof_metrics(
                y_true_log=y_reg_target,
                oof_predictions_log=s2_cv_result.oof_predictions,
            )

            # Add mean CV metrics
            cv_rmse = [m["rmse"] for m in s2_cv_result.fold_metrics]
            cv_mae = [m["mae"] for m in s2_cv_result.fold_metrics]
            s2_metrics["s2_mean_rmse"] = float(np.mean(cv_rmse))
            s2_metrics["s2_mean_mae"] = float(np.mean(cv_mae))

            # Train Final Model
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
                quiet=ctx.quiet,
            )

            # Log results
            logger.log_regressor_results(s2_metrics, s2_model, X_reg, cfg, ctx.quiet)

            # Bubble up key metrics to pipeline run
            mlflow.log_metrics(
                {
                    "s2_oof_rmse": s2_metrics["s2_oof_rmse"],
                    "s2_oof_mae": s2_metrics["s2_oof_mae"],
                }
            )

        if not ctx.quiet:
            print("\nRegressor Training Complete. Model saved to MLflow.")

        available_metrics = {
            "s2_oof_rmse": s2_metrics["s2_oof_rmse"],
            "s2_oof_mae": s2_metrics["s2_oof_mae"],
            "s2_mean_rmse": s2_metrics["s2_mean_rmse"],
            "s2_mean_mae": s2_metrics["s2_mean_mae"],
        }

        opt_metric = cfg.optimization.metric
        current_score = available_metrics.get(opt_metric)
        if current_score is None:
            raise ValueError(
                f"Metric '{opt_metric}' is not available. "
                f"Available metrics: {list(available_metrics.keys())}"
            )

        # Update sweep champion
        if ctx.is_sweep:
            is_new_best = sweep_mgr.update_champion(
                ctx, pipeline_run.info.run_id, current_score, cfg
            )
            sweep_mgr.print_leaderboard(ctx, current_score, is_new_best, cfg)

        # Write DVC metrics
        logger.write_dvc_metrics(opt_metric, current_score, "regressor_metrics.json")

        return float(current_score)


if __name__ == "__main__":
    main()
