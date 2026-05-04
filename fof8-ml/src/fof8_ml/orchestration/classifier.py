import joblib
import mlflow
import numpy as np

from fof8_ml.models.calibration import BetaCalibrator, run_calibration_audit
from fof8_ml.orchestration.evaluator import compute_classifier_final_metrics, optimize_threshold
from fof8_ml.orchestration.pipeline_runner import PipelineContext
from fof8_ml.orchestration.pipeline_types import ClassifierResult
from fof8_ml.orchestration.trainer import run_cv_classifier, train_final_model


def run_classifier(ctx: PipelineContext) -> dict[str, float]:
    cfg = ctx.cfg
    data = ctx.data
    quiet = ctx.sweep_context.quiet

    if not quiet:
        print("\n" + "=" * 40)
        print("CLASSIFIER: SIEVE CLASSIFIER")
        print("=" * 40)

    with ctx.logger.start_model_run("classifier", ctx.sweep_context):
        ctx.logger.log_model_params(cfg.model, prefix="classifier")
        mlflow.log_param("target.classifier.target_col", cfg.target.classifier_sieve.target_col)

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
        mlflow.log_metrics({f"classifier_pre_audit_{k}": v for k, v in pre_audit_results.items()})

        if not quiet:
            print("\nFitting Beta Calibrator...")
        calibrator = BetaCalibrator()
        calibrator.fit(cv_result.oof_predictions, data.y_cls)
        calibrator_path = "classifier_beta_calibrator.joblib"
        joblib.dump(calibrator, calibrator_path)
        if not quiet:
            mlflow.log_artifact(calibrator_path)

        calibrated_oof_probs = calibrator.predict(cv_result.oof_predictions)

        audit_results = run_calibration_audit(data.y_cls, calibrated_oof_probs)
        mlflow.log_metrics({f"classifier_audit_{k}": v for k, v in audit_results.items()})
        if not quiet:
            print(
                f"Calibration Audit: Intercept={audit_results['cox_intercept']:.4f}, "
                f"Slope={audit_results['cox_slope']:.4f}, "
                f"p={audit_results['spiegelhalter_p']:.4f}"
            )

        if not quiet:
            target_recall = cfg.target.classifier_sieve.min_positive_recall
            print(
                f"\nOptimizing Classifier Threshold (Calibrated) (Constraint: "
                f"Min Positive Recall >= {target_recall})..."
            )
        best_threshold, _ = optimize_threshold(
            y_true=data.y_cls,
            calibrated_probs=calibrated_oof_probs,
            min_positive_recall=cfg.target.classifier_sieve.min_positive_recall,
        )

        classifier_metrics = compute_classifier_final_metrics(
            y_true=data.y_cls,
            calibrated_probs=calibrated_oof_probs,
            threshold=best_threshold,
        )

        final_preds = (calibrated_oof_probs >= best_threshold).astype(int)

        classifier_result = ClassifierResult(
            cv_result=cv_result,
            calibrated_oof_probs=calibrated_oof_probs,
            raw_oof_probs=cv_result.oof_predictions,
            optimal_threshold=best_threshold,
            final_predictions=final_preds,
            metrics=classifier_metrics,
            calibrator=calibrator,
        )

        avg_best_iters = int(np.mean(cv_result.best_iterations))
        classifier_model = train_final_model(
            model_cfg=cfg.model,
            role="classifier",
            X=data.X_train,
            y=data.y_cls,
            avg_best_iterations=avg_best_iters,
            seed=cfg.seed,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
        )

        ctx.logger.log_classifier_results(classifier_result, classifier_model, data, cfg, quiet)

        mlflow.log_metrics(
            {
                "classifier_oof_f1_bust": classifier_metrics["classifier_oof_f1_bust"],
                "classifier_oof_hit_recall": classifier_metrics["classifier_oof_hit_recall"],
                "classifier_optimal_threshold": best_threshold,
                "classifier_oof_pr_auc": classifier_metrics["classifier_oof_pr_auc"],
            }
        )

    return {
        "classifier_oof_f1_bust": classifier_metrics["classifier_oof_f1_bust"],
        "classifier_oof_recall_bust": classifier_metrics["classifier_oof_recall_bust"],
        "classifier_oof_pr_auc": classifier_metrics["classifier_oof_pr_auc"],
    }
