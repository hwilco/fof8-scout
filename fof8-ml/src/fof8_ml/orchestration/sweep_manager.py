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

            cfg_container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            champ_params = {
                k: v for k, v in cfg_container.items() if k in ["stage1_model", "stage2_model"]
            }
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

            if cfg.train_stage2 and cfg.stage2_model is not None:
                mlflow.register_model(
                    model_uri=f"runs:/{pipeline_run_id}/stage2_model",
                    name="fof8-scout-regressor",
                )

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
            cfg_container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            champ_params = {
                k: v for k, v in cfg_container.items() if k in ["stage1_model", "stage2_model"]
            }
            print(f"NEW CHAMPION PARAMS: {champ_params}")
        print("=" * 60 + "\n")
