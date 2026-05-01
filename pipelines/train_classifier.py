import logging
import os
import warnings

import hydra
import joblib
import matplotlib
import mlflow
import numpy as np
from fof8_ml.models.calibration import BetaCalibrator, run_calibration_audit
from fof8_ml.orchestration import (
    DataLoader,
    ExperimentLogger,
    Stage1Result,
    SweepManager,
    compute_stage1_final_metrics,
    optimize_threshold,
    run_cv_classifier,
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


@hydra.main(version_base=None, config_path="conf", config_name="classifier_pipeline")
def main(cfg: DictConfig) -> float:
    """Train and evaluate the Stage 1 Sieve Classifier.

    Args:
        cfg: Hydra config from classifier_pipeline.yaml.

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

    # 3. Train & Evaluate
    with logger.start_pipeline_run(f"Classifier_{cfg.model.name}", tags=ctx.tags) as pipeline_run:
        logger.log_data_summary(data, cfg, absolute_raw_path, ctx.is_sweep, ctx.trial_num)

        if not ctx.quiet:
            print("\n" + "=" * 40)
            print("STAGE 1: SIEVE CLASSIFIER")
            print("=" * 40)

        with logger.start_stage_run("stage1", ctx):
            logger.log_stage_params(cfg.model, prefix="s1")

            # Cross-validation
            cv_result = run_cv_classifier(
                X=data.X_train,
                y=data.y_cls,
                model_cfg=cfg.model,
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
                target_recall = cfg.target.stage1_sieve.min_survivor_recall
                print(
                    f"\nOptimizing Stage 1 Threshold (Calibrated) (Constraint: "
                    f"Min Survivor Recall >= {target_recall})..."
                )
            best_threshold, best_f1_0 = optimize_threshold(
                y_true=data.y_cls,
                calibrated_probs=calibrated_oof_probs,
                min_survivor_recall=cfg.target.stage1_sieve.min_survivor_recall,
            )

            # Final Metrics
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

            # Train Final Model
            avg_best_iters = int(np.mean(cv_result.best_iterations))
            s1_model = train_final_model(
                model_cfg=cfg.model,
                stage="stage1",
                X=data.X_train,
                y=data.y_cls,
                avg_best_iterations=avg_best_iters,
                seed=cfg.seed,
                use_gpu=cfg.use_gpu,
                quiet=ctx.quiet,
            )

            # Log results
            logger.log_classifier_results(s1_result, s1_model, data, cfg, ctx.quiet)

            # Bubble up key metrics to pipeline run
            mlflow.log_metrics(
                {
                    "s1_oof_f1_bust": s1_metrics["s1_oof_f1_bust"],
                    "s1_oof_hit_recall": s1_metrics["s1_oof_hit_recall"],
                    "s1_optimal_threshold": best_threshold,
                    "s1_oof_pr_auc": s1_metrics["s1_oof_pr_auc"],
                }
            )

        available_metrics = {
            "s1_oof_f1_bust": s1_metrics["s1_oof_f1_bust"],
            "s1_oof_recall_bust": s1_metrics["s1_oof_recall_bust"],
            "s1_oof_pr_auc": s1_metrics["s1_oof_pr_auc"],
        }

        if not ctx.quiet:
            print("\nClassifier Training Complete. Model saved to MLflow.")

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
        logger.write_dvc_metrics(opt_metric, current_score, "classifier_metrics.json")

        return float(current_score)


if __name__ == "__main__":
    main()
