import joblib
import mlflow
import numpy as np

from fof8_ml.models.calibration import BetaCalibrator, run_calibration_audit
from fof8_ml.orchestration.evaluator import compute_stage1_final_metrics, optimize_threshold
from fof8_ml.orchestration.pipeline_runner import PipelineContext
from fof8_ml.orchestration.pipeline_types import Stage1Result
from fof8_ml.orchestration.trainer import run_cv_classifier, train_final_model


def run_classifier_stage(ctx: PipelineContext) -> dict[str, float]:
    cfg = ctx.cfg
    data = ctx.data
    quiet = ctx.sweep_context.quiet

    if not quiet:
        print("\n" + "=" * 40)
        print("STAGE 1: SIEVE CLASSIFIER")
        print("=" * 40)

    with ctx.logger.start_stage_run("stage1", ctx.sweep_context):
        ctx.logger.log_stage_params(cfg.model, prefix="s1")

        cv_result = run_cv_classifier(
            X=data.X_train,
            y=data.y_cls,
            model_cfg=cfg.model,
            cv_cfg=cfg.cv,
            seed=cfg.seed,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
        )

        pre_audit_results = run_calibration_audit(data.y_cls, cv_result.oof_predictions)
        mlflow.log_metrics({f"s1_pre_audit_{k}": v for k, v in pre_audit_results.items()})

        if not quiet:
            print("\nFitting Beta Calibrator...")
        calibrator = BetaCalibrator()
        calibrator.fit(cv_result.oof_predictions, data.y_cls)
        calibrator_path = "stage1_beta_calibrator.joblib"
        joblib.dump(calibrator, calibrator_path)
        if not quiet:
            mlflow.log_artifact(calibrator_path)

        calibrated_oof_probs = calibrator.predict(cv_result.oof_predictions)

        audit_results = run_calibration_audit(data.y_cls, calibrated_oof_probs)
        mlflow.log_metrics({f"s1_audit_{k}": v for k, v in audit_results.items()})
        if not quiet:
            print(
                f"Calibration Audit: Intercept={audit_results['cox_intercept']:.4f}, "
                f"Slope={audit_results['cox_slope']:.4f}, "
                f"p={audit_results['spiegelhalter_p']:.4f}"
            )

        if not quiet:
            target_recall = cfg.target.stage1_sieve.min_positive_recall
            print(
                f"\nOptimizing Stage 1 Threshold (Calibrated) (Constraint: "
                f"Min Positive Recall >= {target_recall})..."
            )
        best_threshold, _ = optimize_threshold(
            y_true=data.y_cls,
            calibrated_probs=calibrated_oof_probs,
            min_positive_recall=cfg.target.stage1_sieve.min_positive_recall,
        )

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

        avg_best_iters = int(np.mean(cv_result.best_iterations))
        s1_model = train_final_model(
            model_cfg=cfg.model,
            stage="stage1",
            X=data.X_train,
            y=data.y_cls,
            avg_best_iterations=avg_best_iters,
            seed=cfg.seed,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
        )

        ctx.logger.log_classifier_results(s1_result, s1_model, data, cfg, quiet)

        mlflow.log_metrics(
            {
                "s1_oof_f1_bust": s1_metrics["s1_oof_f1_bust"],
                "s1_oof_hit_recall": s1_metrics["s1_oof_hit_recall"],
                "s1_optimal_threshold": best_threshold,
                "s1_oof_pr_auc": s1_metrics["s1_oof_pr_auc"],
            }
        )

    return {
        "s1_oof_f1_bust": s1_metrics["s1_oof_f1_bust"],
        "s1_oof_recall_bust": s1_metrics["s1_oof_recall_bust"],
        "s1_oof_pr_auc": s1_metrics["s1_oof_pr_auc"],
    }
