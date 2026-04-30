import logging
import os
import warnings

import hydra
import joblib
import matplotlib
import numpy as np
import polars as pl
from fof8_ml.models.calibration import BetaCalibrator, run_calibration_audit
from fof8_ml.orchestration import (
    DataLoader,
    ExperimentLogger,
    Stage1Result,
    SweepManager,
    compute_stage1_final_metrics,
    compute_stage2_oof_metrics,
    optimize_threshold,
    run_cv_classifier,
    run_cv_regressor,
    train_final_model,
)
from hydra.core.hydra_config import HydraConfig
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


@hydra.main(version_base=None, config_path="conf", config_name="economic_pipeline")
def main(cfg: DictConfig) -> float:
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

    # Always print data summary unless quiet sweep
    if not ctx.quiet:
        print(f"Simulation Range: {data.timeline.initial_year} to {data.timeline.final_sim_year}")
        print(
            f"Active Range: {data.timeline.valid_start_year} to {data.timeline.valid_end_year} "
            f"(Buffer: {cfg.split.right_censor_buffer} years)"
        )
        print(
            f"Training Set: {data.timeline.train_year_range} "
            f"({data.timeline.train_year_range[1] - data.timeline.train_year_range[0] + 1} classes)"
        )
        print(
            f"Holdout Set: {data.timeline.test_year_range} "
            f"({data.timeline.test_year_range[1] - data.timeline.test_year_range[0] + 1} classes)"
        )
        if cfg.mask_positional_features:
            print("Applying In-Memory Positional Feature Masking...")

    # Trial-specific feature ablation
    data = loader.apply_feature_ablation(
        data, cfg.get("include_features"), cfg.get("exclude_features")
    )

    # Determine trial number for logging
    trial_num = None
    if ctx.is_sweep:
        try:
            trial_num = HydraConfig.get().job.num
        except Exception:
            pass

    # 3. Train & Evaluate
    s1_name = cfg.stage1_model.name
    s2_name = cfg.stage2_model.name if cfg.stage2_model else "None"

    with logger.start_pipeline_run(f"Pipeline_{s1_name}_{s2_name}", tags=ctx.tags) as pipeline_run:
        logger.log_data_summary(data, cfg, absolute_raw_path, ctx.is_sweep, trial_num)

        # ---------------------------------------------------------
        # STAGE 1: SIEVE CLASSIFIER
        # ---------------------------------------------------------
        stage1_run_id = cfg.get("stage1_run_id")

        if not stage1_run_id:
            if not ctx.quiet:
                print("\n" + "=" * 40)
                print("STAGE 1: SIEVE CLASSIFIER")
                print("=" * 40)

            import mlflow

            with mlflow.start_run(run_name="Stage1_Sieve_Classifier", nested=True):
                mlflow.set_tag("model_stage", "stage1")
                if ctx.sweep_name:
                    mlflow.set_tag("sweep_name", ctx.sweep_name)
                if ctx.sweep_run_id:
                    mlflow.set_tag("sweep_run_id", ctx.sweep_run_id)

                # Log Stage 1 specific parameters
                from fof8_ml.orchestration.experiment_logger import flatten_dict, log_params_safe
                from omegaconf import OmegaConf

                s1_params = OmegaConf.to_container(cfg.stage1_model.params, resolve=True)
                log_params_safe(flatten_dict(s1_params, parent_key="s1"))

                # Cross-validation
                cv_result = run_cv_classifier(
                    X=data.X_train,
                    y=data.y_cls,
                    model_cfg=cfg.stage1_model,
                    cv_cfg=cfg.cv,
                    seed=cfg.seed,
                    use_gpu=cfg.use_gpu,
                    quiet=ctx.quiet,
                )

                # Calibration Audit (Pre)
                pre_audit_results = run_calibration_audit(data.y_cls, cv_result.oof_predictions)
                mlflow.log_metrics({f"s1_pre_audit_{k}": v for k, v in pre_audit_results.items()})

                # Fit Calibrator
                if not ctx.quiet:
                    print("\nFitting Beta Calibrator...")
                calibrator = BetaCalibrator()
                calibrator.fit(cv_result.oof_predictions, data.y_cls)
                calibrator_path = "stage1_beta_calibrator.joblib"
                joblib.dump(calibrator, calibrator_path)
                if not ctx.quiet:
                    mlflow.log_artifact(calibrator_path)

                calibrated_oof_probs = calibrator.predict(cv_result.oof_predictions)

                # Calibration Audit (Post)
                audit_results = run_calibration_audit(data.y_cls, calibrated_oof_probs)
                mlflow.log_metrics({f"s1_audit_{k}": v for k, v in audit_results.items()})
                if not ctx.quiet:
                    print(
                        f"Calibration Audit: Intercept={audit_results['cox_intercept']:.4f}, "
                        f"Slope={audit_results['cox_slope']:.4f}, "
                        f"p={audit_results['spiegelhalter_p']:.4f}"
                    )

                # Threshold Optimization
                if not ctx.quiet:
                    print(
                        f"\nOptimizing Stage 1 Threshold (Calibrated) (Constraint: Min Survivor "
                        f"Recall >= {cfg.target.stage1_sieve.min_survivor_recall})..."
                    )
                best_threshold, best_f1_0 = optimize_threshold(
                    y_true=data.y_cls,
                    calibrated_probs=calibrated_oof_probs,
                    min_survivor_recall=cfg.target.stage1_sieve.min_survivor_recall,
                )

                # Final Metrics computation
                s1_metrics = compute_stage1_final_metrics(
                    y_true=data.y_cls,
                    calibrated_probs=calibrated_oof_probs,
                    threshold=best_threshold,
                )

                final_preds = (calibrated_oof_probs >= best_threshold).astype(int)

                s1_result = Stage1Result(
                    cv_result=cv_result,
                    calibrated_oof_probs=calibrated_oof_probs,
                    raw_oof_probs=cv_result.oof_predictions,
                    optimal_threshold=best_threshold,
                    final_predictions=final_preds,
                    metrics=s1_metrics,
                    calibrator=calibrator,
                )

                # Train Final Stage 1 Model
                avg_best_iters = int(np.mean(cv_result.best_iterations))
                s1_model = train_final_model(
                    model_cfg=cfg.stage1_model,
                    stage="stage1",
                    X=data.X_train,
                    y=data.y_cls,
                    avg_best_iterations=avg_best_iters,
                    seed=cfg.seed,
                    use_gpu=cfg.use_gpu,
                    quiet=ctx.quiet,
                )

                # Log everything
                logger.log_stage1_results(s1_result, s1_model, data, cfg, ctx.quiet)

                # Bubble up metrics
                mlflow.log_metrics(
                    {
                        "s1_oof_f1_bust": s1_metrics["s1_oof_f1_bust"],
                        "s1_oof_hit_recall": s1_metrics["s1_oof_hit_recall"],
                        "s1_optimal_threshold": best_threshold,
                        "s1_oof_pr_auc": s1_metrics["s1_oof_pr_auc"],
                    }
                )

                s1_mask = (data.y_cls == 1).astype(bool)
                s1_oof_pr_auc = s1_metrics["s1_oof_pr_auc"]
                best_f1_0 = s1_metrics["s1_oof_f1_bust"]
                bust_recall = s1_metrics["s1_oof_recall_bust"]
        else:
            print(f"\nSKIPPING STAGE 1: Using results from Run {stage1_run_id}")
            import mlflow

            logger.client.download_artifacts(stage1_run_id, "stage1_oof_results.csv")
            s1_mask = (data.y_cls == 1).astype(bool)

            mlflow.set_tag("stage1_source_run", stage1_run_id)
            try:
                best_f1_0 = logger.client.get_run(stage1_run_id).data.metrics.get(
                    "s1_oof_f1_bust", 0.0
                )
                s1_oof_pr_auc = logger.client.get_run(stage1_run_id).data.metrics.get(
                    "s1_oof_pr_auc", 0.0
                )
                bust_recall = logger.client.get_run(stage1_run_id).data.metrics.get(
                    "s1_oof_recall_bust", 0.0
                )
            except Exception:
                best_f1_0 = 0.0
                s1_oof_pr_auc = 0.0
                bust_recall = 0.0

        # ---------------------------------------------------------
        # STAGE 2: INTENSITY REGRESSOR
        # ---------------------------------------------------------
        s2_metrics_dict = {}
        if cfg.train_stage2 and cfg.stage2_model is not None:
            if not ctx.quiet:
                print("\n" + "=" * 40)
                print("STAGE 2: INTENSITY REGRESSOR")
                print("=" * 40)

            with mlflow.start_run(run_name="Stage2_Intensity_Regressor", nested=True):
                mlflow.set_tag("model_stage", "stage2")
                if ctx.sweep_name:
                    mlflow.set_tag("sweep_name", ctx.sweep_name)
                if ctx.sweep_run_id:
                    mlflow.set_tag("sweep_run_id", ctx.sweep_run_id)

                from fof8_ml.orchestration.experiment_logger import flatten_dict, log_params_safe
                from omegaconf import OmegaConf

                s2_params = OmegaConf.to_container(cfg.stage2_model.params, resolve=True)
                log_params_safe(flatten_dict(s2_params, parent_key="s2"))

                X_reg = data.X_train.filter(pl.Series(s1_mask))
                y_reg_target = np.log1p(data.y_reg[s1_mask])

                s2_cv_result = run_cv_regressor(
                    X=X_reg,
                    y=y_reg_target,
                    model_cfg=cfg.stage2_model,
                    cv_cfg=cfg.cv,
                    seed=cfg.seed,
                    use_gpu=cfg.use_gpu,
                    quiet=ctx.quiet,
                )

                s2_metrics_dict = compute_stage2_oof_metrics(
                    y_true_log=y_reg_target,
                    oof_predictions_log=s2_cv_result.oof_predictions,
                )

                # Add mean CV metrics
                cv_rmse = [m["rmse"] for m in s2_cv_result.fold_metrics]
                cv_mae = [m["mae"] for m in s2_cv_result.fold_metrics]
                s2_metrics_dict["s2_mean_rmse"] = float(np.mean(cv_rmse))
                s2_metrics_dict["s2_mean_mae"] = float(np.mean(cv_mae))

                avg_best_iters_reg = (
                    int(np.mean(s2_cv_result.best_iterations))
                    if s2_cv_result.best_iterations
                    else 100
                )
                s2_model = train_final_model(
                    model_cfg=cfg.stage2_model,
                    stage="stage2",
                    X=X_reg,
                    y=y_reg_target,
                    avg_best_iterations=avg_best_iters_reg,
                    seed=cfg.seed,
                    use_gpu=cfg.use_gpu,
                    quiet=ctx.quiet,
                )

                logger.log_stage2_results(s2_metrics_dict, s2_model, X_reg, cfg, ctx.quiet)
                # Bubble up s2 metrics
                mlflow.log_metrics(
                    {
                        "s2_oof_rmse": s2_metrics_dict["s2_oof_rmse"],
                        "s2_oof_mae": s2_metrics_dict["s2_oof_mae"],
                    }
                )

        if not ctx.quiet:
            print("\nFull Pipeline Training Complete. Models saved to MLflow.")

        # --- Metric Consolidation for Optimization/Return ---
        available_metrics = {
            "s1_oof_f1_bust": best_f1_0,
            "s1_oof_recall_bust": bust_recall,
            "s1_oof_pr_auc": s1_oof_pr_auc,
        }
        if "s2_oof_rmse" in s2_metrics_dict:
            available_metrics["s2_oof_rmse"] = s2_metrics_dict["s2_oof_rmse"]

        opt_metric = cfg.optimization.metric
        current_score = available_metrics.get(opt_metric)
        if current_score is None:
            raise ValueError(
                f"Metric '{opt_metric}' is not available for optimization. "
                f"Available metrics: {list(available_metrics.keys())}"
            )

        # --- Update Sweep Parent with Best Results ---
        if ctx.is_sweep:
            is_new_best = sweep_mgr.update_champion(
                ctx, pipeline_run.info.run_id, current_score, cfg
            )
            sweep_mgr.print_leaderboard(ctx, current_score, is_new_best, cfg)

        # --- Write DVC Metrics ---
        logger.write_dvc_metrics(opt_metric, current_score)

        return float(current_score)


if __name__ == "__main__":
    main()
