from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from omegaconf import DictConfig

from fof8_ml.orchestration.data_loader import DataLoader
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.pipeline_types import PreparedData
from fof8_ml.orchestration.sweep_manager import SweepContext, SweepManager


@dataclass
class PipelineContext:
    cfg: DictConfig
    exp_root: str
    absolute_raw_path: str
    logger: ExperimentLogger
    sweep_mgr: SweepManager
    sweep_context: SweepContext
    data: PreparedData


def resolve_exp_root(entrypoint_file: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(entrypoint_file))
    return os.path.abspath(os.path.join(script_dir, ".."))


def build_pipeline_context(cfg: DictConfig, entrypoint_file: str) -> PipelineContext:
    exp_root = resolve_exp_root(entrypoint_file)
    absolute_raw_path = os.path.abspath(os.path.join(exp_root, cfg.data.raw_path))

    logger = ExperimentLogger(exp_root, cfg.experiment_name)
    logger.init_tracking()
    if logger.client is None or logger.experiment_id is None:
        raise RuntimeError("MLflow tracking client/experiment were not initialized")

    sweep_mgr = SweepManager(logger.client, logger.experiment_id, exp_root)
    sweep_context = sweep_mgr.detect_sweep(cfg)

    loader = DataLoader(exp_root, quiet=sweep_context.quiet)
    data = loader.load(cfg)
    loader.print_summary(data, cfg)
    data = loader.apply_feature_ablation(
        data, cfg.get("include_features"), cfg.get("exclude_features")
    )

    return PipelineContext(
        cfg=cfg,
        exp_root=exp_root,
        absolute_raw_path=absolute_raw_path,
        logger=logger,
        sweep_mgr=sweep_mgr,
        sweep_context=sweep_context,
        data=data,
    )


def select_optimization_metric(available_metrics: Mapping[str, float], metric_name: str) -> float:
    current_score = available_metrics.get(metric_name)
    if current_score is None:
        raise ValueError(
            f"Metric '{metric_name}' is not available. "
            f"Available metrics: {list(available_metrics.keys())}"
        )
    return float(current_score)


def finalize_pipeline_run(
    ctx: PipelineContext,
    pipeline_run_id: str,
    metric_name: str,
    current_score: float,
    metrics_filename: str,
) -> float:
    if ctx.sweep_context.is_sweep:
        is_new_best = ctx.sweep_mgr.update_champion(
            ctx.sweep_context, pipeline_run_id, current_score, ctx.cfg
        )
        ctx.sweep_mgr.print_leaderboard(ctx.sweep_context, current_score, is_new_best, ctx.cfg)

    ctx.logger.write_dvc_metrics(metric_name, current_score, metrics_filename)
    return float(current_score)
