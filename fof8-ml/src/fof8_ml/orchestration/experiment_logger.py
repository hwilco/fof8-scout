import contextlib
import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional

import dagshub
import dvc.api
import mlflow
from omegaconf import DictConfig, OmegaConf

from fof8_ml.evaluation.plotting import (
    log_calibration_comparison,
    log_confusion_matrix,
    log_feature_importance,
)
from fof8_ml.models.base import ModelWrapper
from fof8_ml.orchestration.pipeline_types import PreparedData, Stage1Result

# Global tracking flags
_TRACKING_INITIALIZED = False
_USING_REMOTE = None


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_params_safe(params: dict):
    """Log parameters to MLflow in chunks of 100 to avoid limits."""
    items = list(params.items())
    for i in range(0, len(items), 100):
        mlflow.log_params(dict(items[i : i + 100]))


@contextlib.contextmanager
def preserve_cwd(new_cwd: str = None):
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

    def __init__(self, exp_root: str, experiment_name: str):
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
        self.experiment_id = exp.experiment_id

    def start_pipeline_run(self, run_name: str, tags: dict) -> mlflow.ActiveRun:
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
        mlflow.set_tag("data.league", cfg.data.league_name)
        mlflow.set_tag("data.raw_path", cfg.data.raw_path)

        try:
            git_commit = self.get_git_commit()
            mlflow.set_tag("git_commit", git_commit)
            with preserve_cwd(self.exp_root):
                data_url = dvc.api.get_url(path=cfg.data.raw_path, remote="origin")
            mlflow.set_tag("dvc.data_url", data_url)
        except Exception as e:
            logging.warning(f"Could not log Git commit / DVC version: {e}")

        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        log_params_safe(flatten_dict(cfg_container))

        if cfg.get("tags"):
            mlflow.set_tags(OmegaConf.to_container(cfg.tags, resolve=True))

        if is_sweep and trial_num is not None:
            mlflow.set_tag("trial_num", str(trial_num))

        n_pos = int(data.y_cls.sum())
        n_neg = len(data.y_cls) - n_pos
        mlflow.log_params({"data.n_pos": n_pos, "data.n_neg": n_neg})

    def log_classifier_results(
        self,
        result: Stage1Result,
        model: ModelWrapper,
        data: PreparedData,
        cfg: DictConfig,
        quiet: bool,
    ) -> None:
        """Log Stage 1 metrics, artifacts, and optimal threshold."""
        mlflow.log_params({"s1_optimal_threshold": result.optimal_threshold})
        mlflow.log_metrics(result.metrics)

        if not quiet:
            log_calibration_comparison(
                data.y_cls, result.raw_oof_probs, result.calibrated_oof_probs
            )
            log_confusion_matrix(data.y_cls, result.final_predictions, result.optimal_threshold)

            # Export OOF DataFrame
            import polars as pl

            oof_df = data.meta_train.with_columns(
                [
                    pl.Series("y_true", data.y_cls),
                    pl.Series("oof_prob_raw", result.raw_oof_probs),
                    pl.Series("oof_prob", result.calibrated_oof_probs),
                    pl.Series("cleared_sieve", result.final_predictions),
                ]
            )
            oof_df.write_csv("stage1_oof_results.csv")
            mlflow.log_artifact("stage1_oof_results.csv")

            if cfg.diagnostics.log_importance or cfg.diagnostics.log_shap:
                log_feature_importance(
                    model,
                    "Stage 1 Importance",
                    X=data.X_train,
                    log_shap=cfg.diagnostics.log_shap,
                )

        model.log_model("stage1_model")

    def log_regressor_results(
        self,
        metrics: Dict[str, float],
        model: ModelWrapper,
        X_reg: Any,
        cfg: DictConfig,
        quiet: bool,
    ) -> None:
        """Log Stage 2 metrics and artifacts."""
        mlflow.log_metrics(metrics)
        model.log_model("stage2_model")

        if not quiet:
            if cfg.diagnostics.log_importance or cfg.diagnostics.log_shap:
                log_feature_importance(
                    model,
                    "Stage 2 Importance",
                    X=X_reg,
                    log_shap=cfg.diagnostics.log_shap,
                )

    def start_stage_run(self, stage_name: str, ctx: Any) -> mlflow.ActiveRun:
        """Start a nested MLflow run for a pipeline stage."""
        run_name = f"Stage{stage_name.capitalize()}"
        active_run = mlflow.start_run(run_name=run_name, nested=True)
        mlflow.set_tag("model_stage", stage_name.lower())
        if ctx.sweep_name:
            mlflow.set_tag("sweep_name", ctx.sweep_name)
        if ctx.sweep_run_id:
            mlflow.set_tag("sweep_run_id", ctx.sweep_run_id)
        return active_run

    def log_stage_params(self, model_cfg: DictConfig, prefix: str) -> None:
        """Log stage-specific model parameters."""
        params = OmegaConf.to_container(model_cfg.params, resolve=True)
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
                Defaults to 'metrics.json'. Use stage-specific names (e.g.
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
