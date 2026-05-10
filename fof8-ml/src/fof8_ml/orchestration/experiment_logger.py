import contextlib
import json
import logging
import os
import subprocess
from collections.abc import Iterator
from typing import Any, Optional, cast

import dagshub
import dvc.api
import mlflow
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from fof8_ml.data.schema import FEATURE_SCHEMA_ARTIFACT_PATH, FeatureSchema
from fof8_ml.evaluation.plotting import (
    log_calibration_comparison,
    log_confusion_matrix,
    log_feature_importance,
)
from fof8_ml.models.base import ModelRole, ModelWrapper
from fof8_ml.orchestration.pipeline_types import ClassifierResult, PreparedData

# Global tracking flags
_TRACKING_INITIALIZED = False
_USING_REMOTE = None

ROLE_RUN_NAMES: dict[ModelRole, str] = {
    "classifier": "Classifier",
    "regressor": "Regressor",
}


def resolve_model_role_name(role_name: ModelRole | str) -> str:
    """Return the canonical display name for a model role."""
    normalized_role = role_name.strip().lower()
    if normalized_role not in ROLE_RUN_NAMES:
        raise ValueError(
            f"Unsupported model role '{role_name}'. "
            f"Expected one of: {sorted(ROLE_RUN_NAMES.keys())}"
        )
    return ROLE_RUN_NAMES[cast(ModelRole, normalized_role)]


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, object]:
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_params_safe(params: dict[str, object]) -> None:
    """Log parameters to MLflow in chunks of 100 to avoid limits."""
    items = list(params.items())
    for i in range(0, len(items), 100):
        mlflow.log_params(dict(items[i : i + 100]))


@contextlib.contextmanager
def preserve_cwd(new_cwd: str | None = None) -> Iterator[None]:
    """Temporarily change the working directory."""
    old_cwd = os.getcwd()
    if new_cwd:
        os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class ExperimentLogger:
    """Wraps all MLflow tracking operations with DagsHub fallback."""

    def __init__(self, exp_root: str, experiment_name: str) -> None:
        self.exp_root = exp_root
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
        self.git_commit: Optional[str] = None

    def init_tracking(self) -> None:
        """Initialize MLflow tracking (DagsHub remote with local SQLite fallback)."""
        global _TRACKING_INITIALIZED, _USING_REMOTE
        if not _TRACKING_INITIALIZED:
            if not os.environ.get("MLFLOW_TRACKING_URI"):
                dagshub.init(repo_owner="hwilco", repo_name="fof8-scout", mlflow=True)
                _USING_REMOTE = True
            mlflow.autolog(log_models=False)
            _TRACKING_INITIALIZED = True

        self.client = mlflow.tracking.MlflowClient()
        exp = self.client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            try:
                self.client.create_experiment(self.experiment_name)
            except Exception:
                pass  # Race condition
        mlflow.set_experiment(self.experiment_name)
        exp = self.client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            raise RuntimeError(f"Failed to resolve MLflow experiment '{self.experiment_name}'")
        self.experiment_id = exp.experiment_id

    def start_pipeline_run(self, run_name: str, tags: dict[str, str]) -> mlflow.ActiveRun:
        """Start a top-level run with automatic write-failure fallback."""
        global _USING_REMOTE
        try:
            return mlflow.start_run(run_name=run_name, tags=tags)
        except Exception as e:
            if _USING_REMOTE:
                logging.warning(
                    f">>> Remote MLflow rejected run creation ({e}). "
                    "Falling back to local SQLite storage for this trial."
                )
                db_path = os.path.abspath(os.path.join(self.exp_root, "fof8-ml", "mlflow.db"))
                mlflow.set_tracking_uri(f"sqlite:///{db_path}")
                _USING_REMOTE = False

                local_client = mlflow.tracking.MlflowClient()
                if local_client.get_experiment_by_name(self.experiment_name) is None:
                    local_client.create_experiment(self.experiment_name)
                mlflow.set_experiment(self.experiment_name)

                self.client = local_client
                return mlflow.start_run(run_name=run_name, tags=tags)
            raise

    def get_git_commit(self) -> str:
        """Get or compute git commit."""
        if not self.git_commit:
            with preserve_cwd(self.exp_root):
                try:
                    self.git_commit = (
                        subprocess.check_output(["git", "rev-parse", "HEAD"])
                        .decode("utf-8")
                        .strip()
                    )
                except Exception:
                    self.git_commit = "unknown"
        return self.git_commit

    def log_data_summary(
        self,
        data: PreparedData,
        cfg: DictConfig,
        absolute_raw_path: str,
        is_sweep: bool,
        trial_num: Optional[int],
    ) -> None:
        """Log all data/config parameters and Git/DVC metadata."""
        league_names = cfg.data.get("league_names") or [cfg.data.get("league_name")]
        mlflow.set_tag("data.leagues", ",".join(str(v) for v in league_names if v))
        mlflow.set_tag("data.raw_path", cfg.data.raw_path)

        try:
            git_commit = self.get_git_commit()
            mlflow.set_tag("git_commit", git_commit)
            with preserve_cwd(self.exp_root):
                data_url = dvc.api.get_url(path=cfg.data.raw_path, remote="origin")
            mlflow.set_tag("dvc.data_url", data_url)
        except Exception as e:
            logging.warning(f"Could not log Git commit / DVC version: {e}")

        cfg_container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        log_params_safe(flatten_dict(cfg_container))

        if cfg.get("tags"):
            tags = cast(dict[str, Any], OmegaConf.to_container(cfg.tags, resolve=True))
            mlflow.set_tags(tags)

        if cfg.get("ablation_signature"):
            mlflow.set_tag("ablation.signature", str(cfg.ablation_signature))
        if cfg.get("ablation_enabled_toggles"):
            toggles = [str(t) for t in list(cfg.ablation_enabled_toggles)]
            mlflow.set_tag(
                "ablation.enabled_toggles", ",".join(sorted(toggles)) if toggles else "none"
            )

        if is_sweep and trial_num is not None:
            mlflow.set_tag("trial_num", str(trial_num))

        n_pos = int(data.y_cls.sum())
        n_neg = len(data.y_cls) - n_pos
        mlflow.log_params(
            {
                "data.n_pos": n_pos,
                "data.n_neg": n_neg,
                "data.n_train_rows": len(data.X_train),
                "data.n_val_rows": len(data.X_val),
                "data.n_test_rows": len(data.X_test),
            }
        )

    def log_feature_schema(self, data: PreparedData) -> None:
        """Persist train/inference feature schema to `feature_schema.json`.

        The artifact is logged at the active run artifact root and is loaded by
        `pipelines/batch_inference.py` to enforce training-compatible features.
        """
        schema = FeatureSchema(
            feature_columns=data.X_train.columns,
            categorical_columns=FeatureSchema.infer_categorical_columns(data.X_train),
            excluded_metadata_columns=data.metadata_columns,
            excluded_target_columns=data.target_columns,
            category_handling_policy="cast_to_string",
            missing_column_policy="reject",
            extra_column_policy="drop",
            missing_defaults={},
            college_bucketing_policy={"status": "already_applied_in_feature_pipeline"},
        )
        mlflow.log_dict(schema.to_dict(), FEATURE_SCHEMA_ARTIFACT_PATH)

    def log_classifier_results(
        self,
        result: ClassifierResult,
        model: ModelWrapper,
        data: PreparedData,
        cfg: DictConfig,
        quiet: bool,
        eval_split_label: str = "oof",
        test_calibrated_probs: np.ndarray | None = None,
        test_raw_probs: np.ndarray | None = None,
        test_threshold: float | None = None,
    ) -> None:
        """Log Classifier metrics, artifacts, and optimal threshold."""
        mlflow.log_params({"classifier_optimal_threshold": result.optimal_threshold})
        mlflow.log_metrics(result.metrics)

        if not quiet:
            log_calibration_comparison(
                data.y_cls if eval_split_label == "oof" else data.y_cls_val,
                result.raw_oof_probs,
                result.calibrated_oof_probs,
                eval_label=eval_split_label,
            )
            log_confusion_matrix(
                data.y_cls if eval_split_label == "oof" else data.y_cls_val,
                result.final_predictions,
                result.optimal_threshold,
                eval_label=eval_split_label,
            )

            # Export evaluation-split DataFrame.
            import polars as pl

            eval_meta = data.meta_train if eval_split_label == "oof" else data.meta_val
            eval_y = data.y_cls if eval_split_label == "oof" else data.y_cls_val
            eval_df = eval_meta.with_columns(
                [
                    pl.Series("y_true", eval_y),
                    pl.Series(f"{eval_split_label}_prob_raw", result.raw_oof_probs),
                    pl.Series(f"{eval_split_label}_prob", result.calibrated_oof_probs),
                    pl.Series("cleared_sieve", result.final_predictions),
                ]
            )
            eval_results_path = f"classifier_{eval_split_label}_results.csv"
            eval_df.write_csv(eval_results_path)
            mlflow.log_artifact(eval_results_path)

            if (
                test_calibrated_probs is not None
                and test_raw_probs is not None
                and test_threshold is not None
            ):
                test_preds = (test_calibrated_probs >= test_threshold).astype(int)
                test_df = data.meta_test.with_columns(
                    [
                        pl.Series("y_true", data.y_cls_test),
                        pl.Series("test_prob_raw", test_raw_probs),
                        pl.Series("test_prob", test_calibrated_probs),
                        pl.Series("cleared_sieve", test_preds),
                    ]
                )
                test_df.write_csv("classifier_test_results.csv")
                mlflow.log_artifact("classifier_test_results.csv")

            if cfg.diagnostics.log_importance or cfg.diagnostics.log_shap:
                log_feature_importance(
                    model,
                    "Classifier Importance",
                    X=data.X_train,
                    log_shap=cfg.diagnostics.log_shap,
                )

        model.log_model("classifier_model", X=data.X_train)

    def log_regressor_results(
        self,
        metrics: dict[str, float],
        model: ModelWrapper,
        X_reg: pl.DataFrame,
        cfg: DictConfig,
        quiet: bool,
        test_predictions: np.ndarray | None = None,
        test_meta: pl.DataFrame | None = None,
        test_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log Regressor metrics and artifacts."""
        mlflow.log_metrics(metrics)
        if test_metrics is not None:
            mlflow.log_metrics(test_metrics)
        model.log_model("regressor_model", X=X_reg)

        if not quiet:
            if test_predictions is not None and test_meta is not None:
                test_df = test_meta.with_columns(pl.Series("test_prediction", test_predictions))
                test_df.write_csv("regressor_test_results.csv")
                mlflow.log_artifact("regressor_test_results.csv")

            if cfg.diagnostics.log_importance or cfg.diagnostics.log_shap:
                log_feature_importance(
                    model,
                    "Regressor Importance",
                    X=X_reg,
                    log_shap=cfg.diagnostics.log_shap,
                )

    @contextlib.contextmanager
    def start_model_run(self, role_name: str, ctx: object) -> Iterator[mlflow.ActiveRun | None]:
        """Tag the active pipeline run for a model role without opening a nested run."""
        normalized_role = cast(ModelRole, role_name.strip().lower())
        resolve_model_role_name(normalized_role)
        active_run = mlflow.active_run()
        if active_run is None:
            with mlflow.start_run(run_name=ROLE_RUN_NAMES[normalized_role]) as run:
                mlflow.set_tag("model_role", normalized_role)
                yield run
            return

        mlflow.set_tag("model_role", normalized_role)
        mlflow.set_tag("model_run_layout", "flat")
        sweep_name = getattr(ctx, "sweep_name", None)
        sweep_run_id = getattr(ctx, "sweep_run_id", None)
        if sweep_name:
            mlflow.set_tag("sweep_name", str(sweep_name))
        if sweep_run_id:
            mlflow.set_tag("sweep_run_id", str(sweep_run_id))
        yield active_run

    def log_model_params(self, model_cfg: DictConfig, prefix: str) -> None:
        """Log role-specific model parameters."""
        params = cast(dict[str, Any], OmegaConf.to_container(model_cfg.params, resolve=True))
        log_params_safe(flatten_dict(params, parent_key=prefix))

    def write_dvc_metrics(
        self,
        metric_name: str,
        score: float,
        metrics_filename: str = "metrics.json",
    ) -> None:
        """Write the optimization metric to a JSON file in outputs/ for DVC.

        Args:
            metric_name: Name of the metric being written.
            score: Numeric value of the metric.
            metrics_filename: Output filename inside the outputs/ directory.
                Defaults to 'metrics.json'. Use role-specific names (e.g.
                'classifier_metrics.json') when running independent pipelines.
        """
        try:
            out_dir = os.path.join(self.exp_root, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            metrics_file = os.path.join(out_dir, metrics_filename)

            export_score = float(score) if score is not None else 0.0

            with open(metrics_file, "w") as f:
                json.dump({metric_name: export_score}, f)

            print(f"\n[DVC] Metrics successfully written to: {metrics_file}")
            print(f"[DVC] Final {metric_name.upper()}: {export_score:.4f}")
        except Exception as e:
            print(f"\n[WARNING] Could not write DVC metrics.json: {e}")

    def write_dvc_json(
        self,
        payload: dict[str, object],
        filename: str,
    ) -> None:
        """Write an arbitrary JSON payload to the outputs/ directory for DVC."""
        try:
            out_dir = os.path.join(self.exp_root, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, filename)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

            print(f"\n[DVC] JSON successfully written to: {output_file}")
        except Exception as e:
            print(f"\n[WARNING] Could not write DVC JSON '{filename}': {e}")
