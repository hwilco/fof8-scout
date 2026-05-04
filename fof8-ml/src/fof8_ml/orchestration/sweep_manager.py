import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional, cast

import mlflow
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from fof8_ml.models.registry import get_model_family
from fof8_ml.orchestration.experiment_logger import preserve_cwd


@dataclass
class SweepContext:
    """Captures sweep state for the current trial."""

    is_sweep: bool
    sweep_name: Optional[str]
    sweep_run_id: Optional[str]
    quiet: bool
    tags: dict
    trial_num: Optional[int] = None


class SweepManager:
    """Manages Optuna sweep lifecycle: parent runs, champion tracking, leaderboard."""

    def __init__(
        self, client: mlflow.tracking.MlflowClient, experiment_id: str, exp_root: str
    ) -> None:
        self.client = client
        self.experiment_id = experiment_id
        self.exp_root = exp_root

    def _champion_model_summary(self, cfg: DictConfig) -> dict[str, Any]:
        cfg_container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        model_cfg = cfg_container.get("model")
        if not isinstance(model_cfg, dict):
            return {}

        model_name = model_cfg.get("name")
        model_family = None
        if isinstance(model_name, str):
            for role in ("regressor", "classifier"):
                model_family = get_model_family(role=role, model_name=model_name)
                if model_family is not None:
                    break

        return {
            "model_name": model_name,
            "model_family": model_family,
            "model_params": model_cfg.get("params", {}),
        }

    def _resolve_flat_role_run(self, pipeline_run_id: str, role_name: str) -> Optional[str]:
        """Return the pipeline run when it owns the requested flattened role."""

        normalized_role = role_name.strip().lower()
        try:
            pipeline_run = self.client.get_run(pipeline_run_id)
            if pipeline_run.data.tags.get("model_role") == normalized_role:
                return pipeline_run_id
        except Exception:
            pass
        return None

    def _resolve_model_uri(self, run_id: str, artifact_path: str) -> str:
        """Resolve a model URI across MLflow 2 run artifacts and MLflow 3 logged models."""

        runs_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            artifacts = self.client.list_artifacts(run_id, artifact_path)
            if any(artifact.path == f"{artifact_path}/MLmodel" for artifact in artifacts):
                return runs_uri
        except Exception:
            pass

        run = self.client.get_run(run_id)
        try:
            logged_models = self.client.search_logged_models(
                experiment_ids=[run.info.experiment_id],
                filter_string=f"source_run_id = '{run_id}'",
                max_results=100,
            )
        except Exception:
            logged_models = self.client.search_logged_models(
                experiment_ids=[run.info.experiment_id],
                filter_string=f"name = '{artifact_path}'",
                max_results=100,
            )

        matches = [
            model
            for model in logged_models
            if model.name == artifact_path and model.source_run_id == run_id
        ]
        if not matches:
            return runs_uri

        best_model = max(matches, key=lambda model: model.last_updated_timestamp or 0)
        return f"models:/{best_model.model_id}"

    def _register_role_model(
        self,
        *,
        pipeline_run_id: str,
        role_run_id: str,
        artifact_path: str,
        registered_name: str,
    ) -> None:
        model_uri = self._resolve_model_uri(role_run_id, artifact_path)
        try:
            model_version = mlflow.register_model(model_uri=model_uri, name=registered_name)
        except Exception as exc:
            error = str(exc)[:500]
            self.client.set_tag(pipeline_run_id, "model_registration_error", error)
            raise

        self.client.set_tag(pipeline_run_id, "registered_model_name", registered_name)
        self.client.set_tag(pipeline_run_id, "registered_model_uri", model_uri)
        if getattr(model_version, "version", None) is not None:
            self.client.set_tag(
                pipeline_run_id, "registered_model_version", str(model_version.version)
            )

    def _registration_target(self, cfg: DictConfig) -> tuple[str, str, str] | None:
        """Return role, artifact path, and registry name for champion model registration."""

        model_name = str(cfg.model.name)
        if get_model_family(role="regressor", model_name=model_name) is not None:
            return ("regressor", "regressor_model", "fof8-scout-regressor")
        if get_model_family(role="classifier", model_name=model_name) is not None:
            return ("classifier", "classifier_model", "fof8-scout-classifier")
        return None

    def detect_sweep(self, cfg: DictConfig) -> SweepContext:
        """Detect if we're in a multirun and find/create the parent sweep run."""
        is_sweep = False
        try:
            is_sweep = HydraConfig.get().mode == RunMode.MULTIRUN
            if not is_sweep and "--multirun" not in sys.argv and "-m" not in sys.argv:
                print(f">>> Single Run Mode Detected (Hydra Mode: {HydraConfig.get().mode})")
        except Exception:
            is_sweep = "--multirun" in sys.argv or "-m" in sys.argv

        sweep_run_id = None
        sweep_name = None

        if is_sweep or cfg.get("sweep_name"):
            if cfg.get("sweep_name"):
                sweep_name = cfg.sweep_name
            else:
                try:
                    sweep_name = f"Sweep_{os.path.basename(HydraConfig.get().sweep.dir)}"
                except Exception:
                    sweep_name = "Sweep_Unknown"

            existing_runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.mlflow.runName = '{sweep_name}'",
                max_results=1,
            )
            if existing_runs:
                sweep_run_id = existing_runs[0].info.run_id
            else:
                try:
                    with preserve_cwd(self.exp_root):
                        sweep_git_commit = (
                            subprocess.check_output(["git", "rev-parse", "HEAD"])
                            .decode("utf-8")
                            .strip()
                        )

                    with mlflow.start_run(run_name=sweep_name) as sweep_parent:
                        sweep_run_id = sweep_parent.info.run_id
                        mlflow.set_tag("git_commit", sweep_git_commit)

                        if (
                            "hydra" in cfg
                            and "sweeper" in cfg.hydra
                            and "search_space" in cfg.hydra.sweeper
                        ):
                            search_space = OmegaConf.to_container(
                                cfg.hydra.sweeper.search_space, resolve=True
                            )
                            from fof8_ml.orchestration.experiment_logger import (
                                flatten_dict,
                                log_params_safe,
                            )

                            log_params_safe(
                                flatten_dict(
                                    cast(dict[str, Any], search_space),
                                    parent_key="search_space",
                                )
                            )
                except Exception:
                    existing_runs = self.client.search_runs(
                        experiment_ids=[self.experiment_id],
                        filter_string=f"tags.mlflow.runName = '{sweep_name}'",
                        max_results=1,
                    )
                    if existing_runs:
                        sweep_run_id = existing_runs[0].info.run_id

        tags = {}
        if sweep_run_id:
            tags["mlflow.parentRunId"] = sweep_run_id
            tags["sweep_run_id"] = sweep_run_id
        if sweep_name:
            tags["sweep_name"] = sweep_name

        quiet = bool(is_sweep and cfg.quiet_sweep)

        trial_num = None
        if is_sweep:
            try:
                trial_num = HydraConfig.get().job.num + 1
                n_trials = HydraConfig.get().sweeper.n_trials
                print("\n" + ">" * 10 + f" STARTING TRIAL {trial_num}/{n_trials} " + "<" * 10)
            except Exception:
                pass

        return SweepContext(
            is_sweep=is_sweep,
            sweep_name=sweep_name,
            sweep_run_id=sweep_run_id,
            quiet=quiet,
            tags=tags,
            trial_num=trial_num,
        )

    def update_champion(
        self, ctx: SweepContext, pipeline_run_id: str, current_score: float, cfg: DictConfig
    ) -> bool:
        """Compare against current best, update parent tags if improved. Returns is_new_best."""
        if not ctx.sweep_run_id:
            return False

        opt_metric = cfg.optimization.metric
        higher_is_better = cfg.optimization.direction == "maximize"
        score_name = f"best_{opt_metric}"
        score_label = score_name.replace("best_", "").upper()

        parent_run_data = self.client.get_run(ctx.sweep_run_id).data
        previous_best = parent_run_data.metrics.get(score_name)

        is_new_best = False
        if previous_best is None:
            is_new_best = True
        elif higher_is_better and current_score > previous_best:
            is_new_best = True
        elif not higher_is_better and current_score < previous_best:
            is_new_best = True

        if is_new_best:
            previous_champion_id = parent_run_data.tags.get("best_trial_id")

            self.client.log_metric(ctx.sweep_run_id, score_name, current_score)
            self.client.set_tag(ctx.sweep_run_id, "best_trial_id", pipeline_run_id)

            champ_params = self._champion_model_summary(cfg)
            note_content = (
                f"### 🏆 Current Sweep Champion\n"
                f"- **Run ID:** `{pipeline_run_id}`\n"
                f"- **{score_label}:** {current_score:.4f}\n"
                f"- **Params:** {json.dumps(champ_params, indent=2)}"
            )
            self.client.set_tag(ctx.sweep_run_id, "mlflow.note.content", note_content)

            if previous_champion_id:
                try:
                    self.client.delete_tag(previous_champion_id, "champion")
                except Exception:
                    pass
            self.client.set_tag(pipeline_run_id, "champion", "true")
            self.client.set_tag(ctx.sweep_run_id, "best_params", str(champ_params))

            registration_target = self._registration_target(cfg)
            artifact_path: str | None = None
            registered_name: str | None = None
            if registration_target is not None:
                role_name, artifact_path, registered_name = registration_target
                role_run_id = self._resolve_flat_role_run(pipeline_run_id, role_name)
            else:
                role_run_id = None

            if (
                role_run_id is not None
                and artifact_path is not None
                and registered_name is not None
            ):
                try:
                    self._register_role_model(
                        pipeline_run_id=pipeline_run_id,
                        role_run_id=role_run_id,
                        artifact_path=artifact_path,
                        registered_name=registered_name,
                    )
                except Exception:
                    pass

        return is_new_best

    def print_leaderboard(
        self, ctx: SweepContext, current_score: float, is_new_best: bool, cfg: DictConfig
    ) -> None:
        """Print the sweep leaderboard dashboard to stdout."""
        if not ctx.sweep_run_id:
            return

        opt_metric = cfg.optimization.metric
        score_name = f"best_{opt_metric}"
        score_label = score_name.replace("best_", "").upper()

        try:
            trial_num: int | str = HydraConfig.get().job.num + 1
            n_trials: int | str = HydraConfig.get().sweeper.n_trials
        except Exception:
            trial_num = "?"
            n_trials = "?"

        parent_run = self.client.get_run(ctx.sweep_run_id)
        best_score = parent_run.data.metrics.get(score_name, current_score)
        best_trial_id = parent_run.data.tags.get("best_trial_id")

        best_trial_num = "?"
        if best_trial_id:
            try:
                best_trial_run = self.client.get_run(best_trial_id)
                best_trial_num = best_trial_run.data.tags.get("trial_num", "?")
            except Exception:
                pass

        print("\n" + "=" * 60)
        print(f"🏆 SWEEP LEADERBOARD | Trial [{trial_num}/{n_trials}]")
        print("-" * 60)
        print(
            f"LATEST TRIAL RESULT: {current_score:.4f} ({score_label}) "
            f"({'IMPROVEMENT' if is_new_best else 'No improvement'})"
        )
        print(f"BEST SO FAR:         {best_score:.4f} ({score_label}) [Trial {best_trial_num}]")
        if is_new_best:
            champ_params = self._champion_model_summary(cfg)
            print(f"NEW CHAMPION PARAMS: {champ_params}")
        print("=" * 60 + "\n")
