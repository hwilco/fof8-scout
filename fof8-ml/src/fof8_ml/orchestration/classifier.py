import joblib
import mlflow
import numpy as np
import polars as pl

from fof8_ml.models.calibration import BetaCalibrator, run_calibration_audit
from fof8_ml.orchestration.evaluator import (
    compute_classifier_final_metrics,
    optimize_threshold,
    rename_metric_prefix,
)
from fof8_ml.orchestration.pipeline_runner import PipelineContext
from fof8_ml.orchestration.pipeline_types import ClassifierResult, CVResult
from fof8_ml.orchestration.progress import (
    print_phase,
    print_refit_phase,
    print_test_phase,
    print_validation_run_start,
)
from fof8_ml.orchestration.trainer import (
    run_cv_classifier,
    train_final_model,
    train_model_with_validation,
)


def _concat_features(*frames: pl.DataFrame) -> pl.DataFrame:
    non_empty = [frame for frame in frames if len(frame) > 0]
    return pl.concat(non_empty, how="vertical_relaxed") if non_empty else frames[0].head(0)


def _concat_targets(*arrays: np.ndarray) -> np.ndarray:
    non_empty = [arr for arr in arrays if arr.size > 0]
    return np.concatenate(non_empty) if non_empty else np.array([], dtype=float)


def run_classifier(ctx: PipelineContext) -> dict[str, float]:
    cfg = ctx.cfg
    data = ctx.data
    quiet = ctx.sweep_context.quiet
    progress_every = int(cfg.get("runtime", {}).get("catboost_progress_every", 100))
    refit_final_model = bool(cfg.get("runtime", {}).get("refit_final_model", False))

    if not quiet:
        print("\n" + "=" * 40)
        print("CLASSIFIER: SIEVE CLASSIFIER")
        print("=" * 40)

    with ctx.logger.start_model_run("classifier", ctx.sweep_context):
        ctx.logger.log_model_params(cfg.model, prefix="classifier")
        mlflow.log_param("target.classifier.target_col", cfg.target.classifier_sieve.target_col)
        has_validation = len(data.X_val) > 0
        tuning_model = None
        mlflow.log_param(
            "classifier_mode", "validation_holdout" if has_validation else "grouped_cv"
        )
        mlflow.log_param("classifier_refit_final_model", refit_final_model)

        if has_validation:
            if not quiet:
                print_validation_run_start(
                    role_label="classifier",
                    meta_train=data.meta_train,
                    meta_val=data.meta_val,
                    meta_test=data.meta_test,
                )
                print_phase(
                    "Training classifier on training universes and validating on held-out universes"
                )
            tuning_model = train_model_with_validation(
                model_cfg=cfg.model,
                role="classifier",
                X_train=data.X_train,
                y_train=data.y_cls,
                X_val=data.X_val,
                y_val=data.y_cls_val,
                seed=cfg.seed,
                interactive_progress_every=progress_every,
                use_gpu=cfg.use_gpu,
                quiet=quiet,
            )
            val_raw_probs = tuning_model.predict_proba(data.X_val)
            best_iterations = [tuning_model.get_best_iteration()]
            cv_result = CVResult(
                oof_predictions=val_raw_probs,
                best_iterations=best_iterations,
                fold_metrics=[],
            )
            calibration_y = data.y_cls_val
            calibration_raw_probs = val_raw_probs
        else:
            groups = data.meta_train.get_column("Universe").cast(pl.String).to_numpy()
            cv_result = run_cv_classifier(
                X=data.X_train,
                y=data.y_cls,
                model_cfg=cfg.model,
                cv_cfg=cfg.cv,
                seed=cfg.seed,
                groups=groups,
                interactive_progress_every=progress_every,
                use_gpu=cfg.use_gpu,
                quiet=quiet,
            )
            best_iterations = cv_result.best_iterations
            calibration_y = data.y_cls
            calibration_raw_probs = cv_result.oof_predictions

        pre_audit_results = run_calibration_audit(calibration_y, calibration_raw_probs)
        metric_prefix = "classifier_val_pre_audit_" if has_validation else "classifier_pre_audit_"
        mlflow.log_metrics({f"{metric_prefix}{k}": v for k, v in pre_audit_results.items()})

        if not quiet:
            print_phase("Fitting beta calibrator on validation predictions")
        calibrator = BetaCalibrator()
        calibrator.fit(calibration_raw_probs, calibration_y)
        calibrator_path = "classifier_beta_calibrator.joblib"
        joblib.dump(calibrator, calibrator_path)
        mlflow.log_artifact(calibrator_path)

        calibrated_probs = calibrator.predict(calibration_raw_probs)

        audit_results = run_calibration_audit(calibration_y, calibrated_probs)
        audit_prefix = "classifier_val_audit_" if has_validation else "classifier_audit_"
        mlflow.log_metrics({f"{audit_prefix}{k}": v for k, v in audit_results.items()})
        if not quiet:
            print(
                f"Calibration Audit: Intercept={audit_results['cox_intercept']:.4f}, "
                f"Slope={audit_results['cox_slope']:.4f}, "
                f"p={audit_results['spiegelhalter_p']:.4f}"
            )

        if not quiet:
            target_recall = cfg.target.classifier_sieve.min_positive_recall
            print_phase(
                "Optimizing calibrated classifier threshold "
                f"(min positive recall >= {target_recall})"
            )
        best_threshold, _ = optimize_threshold(
            y_true=calibration_y,
            calibrated_probs=calibrated_probs,
            min_positive_recall=cfg.target.classifier_sieve.min_positive_recall,
        )

        train_metrics = compute_classifier_final_metrics(
            y_true=calibration_y,
            calibrated_probs=calibrated_probs,
            threshold=best_threshold,
        )
        classifier_metrics = rename_metric_prefix(
            train_metrics,
            "classifier_oof_",
            "classifier_val_" if has_validation else "classifier_oof_",
        )
        final_preds = (calibrated_probs >= best_threshold).astype(int)

        classifier_result = ClassifierResult(
            cv_result=cv_result,
            calibrated_oof_probs=calibrated_probs,
            raw_oof_probs=calibration_raw_probs,
            optimal_threshold=best_threshold,
            final_predictions=final_preds,
            metrics=classifier_metrics,
            calibrator=calibrator,
        )

        should_refit = (not has_validation) or refit_final_model
        test_raw_probs = None
        test_calibrated_probs = None
        test_metrics: dict[str, float] = {}

        if should_refit:
            avg_best_iters = int(np.mean(best_iterations)) if best_iterations else 100
            final_train_X = (
                _concat_features(data.X_train, data.X_val) if has_validation else data.X_train
            )
            final_train_y = (
                _concat_targets(data.y_cls, data.y_cls_val) if has_validation else data.y_cls
            )
            if has_validation and not quiet:
                print_refit_phase("classifier", len(data.X_train), len(data.X_val))
            classifier_model = train_final_model(
                model_cfg=cfg.model,
                role="classifier",
                X=final_train_X,
                y=final_train_y,
                avg_best_iterations=avg_best_iters,
                seed=cfg.seed,
                interactive_progress_every=progress_every,
                use_gpu=cfg.use_gpu,
                quiet=quiet,
            )

            if not quiet:
                print_test_phase("classifier", len(data.X_test))
            test_raw_probs = classifier_model.predict_proba(data.X_test)
            test_calibrated_probs = calibrator.predict(test_raw_probs)
            test_metrics = rename_metric_prefix(
                compute_classifier_final_metrics(
                    y_true=data.y_cls_test,
                    calibrated_probs=test_calibrated_probs,
                    threshold=best_threshold,
                ),
                "classifier_oof_",
                "classifier_test_",
            )
            mlflow.log_metrics(test_metrics)
        else:
            assert tuning_model is not None
            classifier_model = tuning_model
            mlflow.log_param("classifier_test_scored", False)

        ctx.logger.log_classifier_results(
            classifier_result,
            classifier_model,
            data,
            cfg,
            quiet,
            eval_split_label="val" if has_validation else "oof",
            test_calibrated_probs=test_calibrated_probs,
            test_raw_probs=test_raw_probs,
            test_threshold=best_threshold,
        )

        mlflow.log_metrics(
            {
                **classifier_metrics,
                "classifier_optimal_threshold": best_threshold,
                **test_metrics,
            }
        )

    primary_prefix = "classifier_val_" if has_validation else "classifier_oof_"
    return {
        f"{primary_prefix}f1_bust": classifier_metrics[f"{primary_prefix}f1_bust"],
        f"{primary_prefix}recall_bust": classifier_metrics[f"{primary_prefix}recall_bust"],
        f"{primary_prefix}pr_auc": classifier_metrics[f"{primary_prefix}pr_auc"],
        "classifier_test_pr_auc": test_metrics.get("classifier_test_pr_auc", 0.0),
    }
