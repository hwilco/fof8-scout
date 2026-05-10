from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf, open_dict

from fof8_ml.orchestration.classifier import run_classifier
from fof8_ml.orchestration.complete_model import (
    resolve_complete_model_inputs,
    run_complete_model_evaluation,
)
from fof8_ml.orchestration.data_loader import DataLoader
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.pipeline_runner import (
    build_pipeline_context,
    resolve_exp_root,
    select_optimization_metric,
)
from fof8_ml.orchestration.regressor import run_regressor
from fof8_ml.orchestration.sweep_manager import SweepManager


@dataclass(frozen=True)
class Phase4Candidate:
    candidate_id: str
    label: str
    regressor_model: str
    regressor_target_col: str
    regressor_target_space: str


@dataclass(frozen=True)
class Phase4RunResult:
    run_id: str
    experiment_name: str
    optimization_metric: str
    optimization_score: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class Phase4CandidateResult:
    candidate_id: str
    label: str
    classifier_source: str
    classifier_run_id: str
    regressor_run_id: str
    complete_run_id: str
    classifier_experiment_name: str
    regressor_experiment_name: str
    complete_experiment_name: str
    classifier_target_col: str
    regressor_target_col: str
    regressor_target_space: str
    regressor_model: str
    regressor_loss_function: str
    ablation_signature: str
    board_artifact_path: str
    position_group_metrics_artifact_path: str
    elite_config: dict[str, Any]
    metrics: dict[str, float]


def _phase4_output_dir(exp_root: str, matrix_name: str, output_subdir: str) -> str:
    return os.path.join(exp_root, "outputs", output_subdir, matrix_name)


def resolve_phase4_candidates(
    phase4_cfg: DictConfig,
    candidate_ids: list[str] | None = None,
) -> list[Phase4Candidate]:
    requested = {candidate_id.strip() for candidate_id in (candidate_ids or []) if candidate_id}
    candidates: list[Phase4Candidate] = []
    for raw_candidate in phase4_cfg.candidates:
        candidate = cast(dict[str, Any], OmegaConf.to_container(raw_candidate, resolve=True))
        candidate_id = str(candidate["candidate_id"])
        if requested and candidate_id not in requested:
            continue
        regressor = cast(dict[str, Any], candidate["regressor"])
        candidates.append(
            Phase4Candidate(
                candidate_id=candidate_id,
                label=str(candidate["label"]),
                regressor_model=str(regressor["model"]),
                regressor_target_col=str(regressor["target_col"]),
                regressor_target_space=str(regressor["target_space"]),
            )
        )
    if requested and not candidates:
        raise ValueError(
            f"No Phase 4 candidates matched candidate_ids={sorted(requested)}. "
            f"Available: {[c.candidate_id for c in resolve_phase4_candidates(phase4_cfg)]}"
        )
    return candidates


def _merge_tags(existing: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = {str(k): v for k, v in existing.items()}
    for key, value in extra.items():
        merged[str(key)] = value
    return merged


def _load_config(exp_root: str, *parts: str) -> DictConfig:
    return OmegaConf.load(os.path.join(exp_root, *parts))


def _load_pipeline_cfg(exp_root: str, pipeline_config_name: str) -> DictConfig:
    root_cfg = _load_config(exp_root, "pipelines", "conf", pipeline_config_name)
    merged_cfg = OmegaConf.create({})
    defaults = list(root_cfg.get("defaults", []))

    for entry in defaults:
        if entry == "_self_":
            merged_cfg = OmegaConf.merge(merged_cfg, root_cfg)
            continue
        if not OmegaConf.is_dict(entry):
            continue

        entry_dict = cast(dict[str, Any], OmegaConf.to_container(entry, resolve=True))
        group_name, option_name = next(iter(entry_dict.items()))
        if group_name.startswith("override "):
            continue
        if option_name is None:
            continue

        group_path = group_name.replace("/", os.sep)
        group_cfg = _load_config(
            exp_root,
            "pipelines",
            "conf",
            group_path,
            f"{option_name}.yaml",
        )
        if group_name == "ablation":
            packaged_group_cfg = group_cfg
        elif group_name == "hparams_search":
            packaged_group_cfg = OmegaConf.create({"hydra": {"sweeper": group_cfg}})
        else:
            packaged_group_cfg = OmegaConf.create({group_name: group_cfg})
        merged_cfg = OmegaConf.merge(merged_cfg, packaged_group_cfg)

    if "_self_" not in defaults:
        merged_cfg = OmegaConf.merge(merged_cfg, root_cfg)

    if "defaults" in merged_cfg:
        with open_dict(merged_cfg):
            del merged_cfg["defaults"]
    return merged_cfg


def _seed_hydra_runtime_defaults(cfg: DictConfig) -> None:
    """Populate minimal Hydra runtime keys needed for interpolation resolution."""

    hydra_cfg = cfg.get("hydra")
    if hydra_cfg is None:
        return

    with open_dict(cfg):
        if cfg.hydra.get("job") is None:
            cfg.hydra.job = OmegaConf.create({})
        if "num" not in cfg.hydra.job:
            cfg.hydra.job.num = 0


def _prepare_pipeline_cfg(
    exp_root: str,
    *,
    pipeline_config_name: str,
    experiment_name: str,
    target_config_name: str,
    model_config_name: str | None,
    runtime_refit_final_model: bool,
    run_tags: dict[str, Any],
    regressor_target_col: str | None = None,
    regressor_target_space: str | None = None,
) -> DictConfig:
    cfg = _load_pipeline_cfg(exp_root, pipeline_config_name)
    target_cfg = _load_config(exp_root, "pipelines", "conf", "target", f"{target_config_name}.yaml")
    with open_dict(cfg):
        cfg.experiment_name = experiment_name
        cfg.target = target_cfg
        cfg.runtime.refit_final_model = runtime_refit_final_model
        cfg.tags = OmegaConf.create(
            _merge_tags(
                cast(dict[str, Any], OmegaConf.to_container(cfg.get("tags", {}), resolve=True)),
                run_tags,
            )
        )
        if model_config_name is not None:
            cfg.model = _load_config(
                exp_root, "pipelines", "conf", "model", f"{model_config_name}.yaml"
            )
        if regressor_target_col is not None:
            cfg.target.regressor_intensity.target_col = regressor_target_col
        if regressor_target_space is not None:
            cfg.target.regressor_intensity.target_space = regressor_target_space
    _seed_hydra_runtime_defaults(cfg)
    return cfg


def _train_classifier_pipeline(exp_root: str, cfg: DictConfig) -> Phase4RunResult:
    entrypoint_file = os.path.join(exp_root, "pipelines", "train_classifier.py")
    ctx = build_pipeline_context(cfg, entrypoint_file)
    with ctx.logger.start_pipeline_run(
        f"Classifier_{cfg.model.name}", tags=ctx.sweep_context.tags
    ) as pipeline_run:
        ctx.logger.log_data_summary(
            ctx.data,
            cfg,
            ctx.absolute_raw_path,
            ctx.sweep_context.is_sweep,
            ctx.sweep_context.trial_num,
        )
        ctx.logger.log_feature_schema(ctx.data)
        metrics = run_classifier(ctx)
        optimization_metric = str(cfg.optimization.metric)
        current_score = select_optimization_metric(metrics, optimization_metric)
        return Phase4RunResult(
            run_id=pipeline_run.info.run_id,
            experiment_name=str(cfg.experiment_name),
            optimization_metric=optimization_metric,
            optimization_score=current_score,
            metrics=metrics,
        )


def _train_regressor_pipeline(exp_root: str, cfg: DictConfig) -> Phase4RunResult:
    entrypoint_file = os.path.join(exp_root, "pipelines", "train_regressor.py")
    ctx = build_pipeline_context(cfg, entrypoint_file)
    with ctx.logger.start_pipeline_run(
        f"Regressor_{cfg.model.name}", tags=ctx.sweep_context.tags
    ) as pipeline_run:
        ctx.logger.log_data_summary(
            ctx.data,
            cfg,
            ctx.absolute_raw_path,
            ctx.sweep_context.is_sweep,
            ctx.sweep_context.trial_num,
        )
        ctx.logger.log_feature_schema(ctx.data)
        metrics = run_regressor(ctx)
        optimization_metric = str(cfg.optimization.metric)
        current_score = select_optimization_metric(metrics, optimization_metric)
        return Phase4RunResult(
            run_id=pipeline_run.info.run_id,
            experiment_name=str(cfg.experiment_name),
            optimization_metric=optimization_metric,
            optimization_score=current_score,
            metrics=metrics,
        )


def _evaluate_complete_model_pipeline(exp_root: str, cfg: DictConfig) -> Phase4RunResult:
    absolute_raw_path = os.path.abspath(os.path.join(exp_root, cfg.data.raw_path))
    logger = ExperimentLogger(exp_root, str(cfg.get("complete_experiment_name")))
    logger.init_tracking()
    if logger.client is None or logger.experiment_id is None:
        raise RuntimeError("MLflow tracking client/experiment were not initialized.")

    inputs = resolve_complete_model_inputs(cfg, exp_root=exp_root, client=logger.client)
    sweep_mgr = SweepManager(logger.client, logger.experiment_id, exp_root)
    sweep_context = sweep_mgr.detect_sweep(cfg)
    data_loader = DataLoader(exp_root, quiet=sweep_context.quiet)
    data = data_loader.load(cfg)

    with logger.start_pipeline_run(
        f"CompleteModel_{inputs.classifier_run_id[:8]}_{inputs.regressor_run_id[:8]}",
        tags={"evaluation_type": "complete_model"},
    ) as pipeline_run:
        logger.log_data_summary(
            data,
            cfg,
            absolute_raw_path,
            sweep_context.is_sweep,
            sweep_context.trial_num,
        )
        metrics = run_complete_model_evaluation(
            cfg=cfg,
            data=data,
            exp_root=exp_root,
            logger=logger,
            inputs=inputs,
        )
        optimization_metric = "complete_draft_value_score"
        current_score = select_optimization_metric(metrics, optimization_metric)
        return Phase4RunResult(
            run_id=pipeline_run.info.run_id,
            experiment_name=str(cfg.get("complete_experiment_name")),
            optimization_metric=optimization_metric,
            optimization_score=current_score,
            metrics=metrics,
        )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_candidate_manifest(
    output_dir: str,
    result: Phase4CandidateResult,
) -> str:
    path = os.path.join(output_dir, f"{result.candidate_id}.json")
    _write_json(path, asdict(result))
    return path


def write_matrix_manifest(
    output_dir: str,
    matrix_name: str,
    manifests: list[dict[str, str]],
) -> str:
    path = os.path.join(output_dir, "matrix_manifest.json")
    _write_json(
        path,
        {
            "matrix_name": matrix_name,
            "candidates": manifests,
        },
    )
    return path


def run_phase4_matrix(cfg: DictConfig, *, exp_root: str | None = None) -> dict[str, Any]:
    exp_root = exp_root or resolve_exp_root(
        os.path.join(os.getcwd(), "pipelines", "run_phase4_matrix.py")
    )
    phase4_cfg = cfg.phase4
    output_dir = _phase4_output_dir(
        exp_root,
        str(phase4_cfg.matrix_name),
        str(phase4_cfg.get("output_subdir", "phase4")),
    )
    os.makedirs(output_dir, exist_ok=True)

    requested_candidate_ids = [str(value) for value in list(cfg.get("candidate_ids", []))]
    candidates = resolve_phase4_candidates(phase4_cfg, requested_candidate_ids)

    shared_tags = cast(dict[str, Any], OmegaConf.to_container(phase4_cfg.shared.tags, resolve=True))
    classifier_source = str(phase4_cfg.shared.classifier_source)
    runtime_refit_final_model = bool(phase4_cfg.shared.runtime.refit_final_model)

    shared_run_tags = {
        **shared_tags,
        "matrix_name": str(phase4_cfg.matrix_name),
        "classifier_source_policy": classifier_source,
    }

    classifier_run_id = None
    classifier_experiment_name = str(phase4_cfg.shared.classifier.experiment_name)
    classifier_target_col = None

    if classifier_source == "fixed_run":
        fixed_run_id = phase4_cfg.shared.get("fixed_classifier_run_id")
        if not fixed_run_id:
            raise ValueError(
                "phase4.shared.fixed_classifier_run_id is required when classifier_source='fixed_run'."
            )
        classifier_run_id = str(fixed_run_id)
        classifier_target_col = str(
            _load_config(
                exp_root,
                "pipelines",
                "conf",
                "target",
                f"{phase4_cfg.shared.classifier.target_config}.yaml",
            ).classifier_sieve.target_col
        )
    elif classifier_source == "train_once_per_matrix":
        classifier_cfg = _prepare_pipeline_cfg(
            exp_root,
            pipeline_config_name="classifier_pipeline.yaml",
            experiment_name=classifier_experiment_name,
            target_config_name=str(phase4_cfg.shared.classifier.target_config),
            model_config_name=str(phase4_cfg.shared.classifier.model),
            runtime_refit_final_model=runtime_refit_final_model,
            run_tags={
                **shared_run_tags,
                "candidate_id": "classifier_shared",
                "candidate_label": "classifier_shared",
            },
        )
        classifier_result = _train_classifier_pipeline(exp_root, classifier_cfg)
        classifier_run_id = classifier_result.run_id
        classifier_target_col = str(classifier_cfg.target.classifier_sieve.target_col)
    else:
        raise ValueError(
            f"Unsupported phase4.shared.classifier_source '{classifier_source}'. "
            "Expected 'fixed_run' or 'train_once_per_matrix'."
        )

    assert classifier_run_id is not None
    assert classifier_target_col is not None

    manifests: list[dict[str, str]] = []
    for candidate in candidates:
        candidate_tags = {
            **shared_run_tags,
            "candidate_id": candidate.candidate_id,
            "candidate_label": candidate.label,
        }
        regressor_cfg = _prepare_pipeline_cfg(
            exp_root,
            pipeline_config_name="regressor_pipeline.yaml",
            experiment_name=str(phase4_cfg.shared.regressor.experiment_name),
            target_config_name=str(phase4_cfg.shared.regressor.target_config),
            model_config_name=candidate.regressor_model,
            runtime_refit_final_model=runtime_refit_final_model,
            run_tags=candidate_tags,
            regressor_target_col=candidate.regressor_target_col,
            regressor_target_space=candidate.regressor_target_space,
        )
        regressor_result = _train_regressor_pipeline(exp_root, regressor_cfg)

        complete_cfg = _load_pipeline_cfg(exp_root, "complete_model_pipeline.yaml")
        with open_dict(complete_cfg):
            complete_cfg.complete_experiment_name = str(
                phase4_cfg.shared.complete_model.experiment_name
            )
            complete_cfg.target = _load_config(
                exp_root,
                "pipelines",
                "conf",
                "target",
                f"{phase4_cfg.shared.complete_model.target_config}.yaml",
            )
            complete_cfg.classifier_run_id = classifier_run_id
            complete_cfg.regressor_run_id = regressor_result.run_id
            complete_cfg.tags = OmegaConf.create(candidate_tags)
        complete_result = _evaluate_complete_model_pipeline(exp_root, complete_cfg)

        elite_cfg = cast(
            dict[str, Any],
            OmegaConf.to_container(
                complete_cfg.target.get("outcome_scorecard", {}).get("elite", {}), resolve=True
            ),
        )
        regressor_model_cfg = cast(
            dict[str, Any], OmegaConf.to_container(regressor_cfg.model, resolve=True)
        )
        regressor_loss_function = str(
            regressor_model_cfg.get("params", {}).get("loss_function", "")
        )
        candidate_result = Phase4CandidateResult(
            candidate_id=candidate.candidate_id,
            label=candidate.label,
            classifier_source=classifier_source,
            classifier_run_id=classifier_run_id,
            regressor_run_id=regressor_result.run_id,
            complete_run_id=complete_result.run_id,
            classifier_experiment_name=classifier_experiment_name,
            regressor_experiment_name=str(phase4_cfg.shared.regressor.experiment_name),
            complete_experiment_name=str(phase4_cfg.shared.complete_model.experiment_name),
            classifier_target_col=classifier_target_col,
            regressor_target_col=candidate.regressor_target_col,
            regressor_target_space=candidate.regressor_target_space,
            regressor_model=candidate.regressor_model,
            regressor_loss_function=regressor_loss_function,
            ablation_signature=str(regressor_cfg.get("ablation_signature", "")),
            board_artifact_path="complete_model_holdout_board.csv",
            position_group_metrics_artifact_path="complete_model_position_group_metrics.csv",
            elite_config=elite_cfg,
            metrics=complete_result.metrics,
        )
        manifest_path = write_candidate_manifest(output_dir, candidate_result)
        manifests.append(
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "manifest_path": manifest_path,
            }
        )

    matrix_manifest_path = write_matrix_manifest(output_dir, str(phase4_cfg.matrix_name), manifests)
    return {
        "matrix_name": str(phase4_cfg.matrix_name),
        "output_dir": output_dir,
        "matrix_manifest_path": matrix_manifest_path,
        "candidate_count": len(manifests),
    }
