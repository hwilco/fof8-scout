from __future__ import annotations

import json
import os
from typing import Any, cast

import polars as pl
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.experiment_matrix import _matrix_output_dir
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root

CORE_METRIC_COLUMNS = [
    "complete_draft_value_score",
    "complete_mean_ndcg_at_64",
    "complete_top64_weighted_mae_normalized",
    "complete_top64_bias",
    "complete_top64_calibration_slope",
    "complete_top64_actual_value",
    "complete_bust_rate_at_32",
    "complete_elite_precision_at_32",
    "complete_elite_recall_at_64",
    "complete_econ_mean_ndcg_at_64",
    "complete_talent_mean_ndcg_at_64",
    "complete_longevity_mean_ndcg_at_64",
]

REGRESSOR_METRIC_COLUMNS = [
    "regressor_val_top32_target_capture_ratio",
    "regressor_val_top64_target_capture_ratio",
    "regressor_val_mean_ndcg_at_32",
    "regressor_val_mean_ndcg_at_64",
    "regressor_val_rmse",
    "regressor_val_mae",
    "regressor_test_draft_value_score",
]


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def load_matrix_manifest(manifest_path: str) -> dict[str, Any]:
    return _read_json(manifest_path)


def load_candidate_manifests(matrix_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _read_json(candidate["manifest_path"])
        for candidate in cast(list[dict[str, Any]], matrix_manifest.get("candidates", []))
    ]


def resolve_matrix_manifest_path(cfg: DictConfig, *, exp_root: str | None = None) -> str:
    if cfg.get("manifest_path"):
        return os.path.abspath(str(cfg.manifest_path))
    exp_root = exp_root or resolve_exp_root(
        os.path.join(os.getcwd(), "pipelines", "export_matrix_report.py")
    )
    return os.path.join(
        _matrix_output_dir(
            exp_root,
            str(cfg.matrix.matrix_name),
            str(cfg.matrix.get("output_subdir", "matrices")),
        ),
        "matrix_manifest.json",
    )


def flatten_candidate_summary(
    client: MlflowClient,
    candidate_manifest: dict[str, Any],
) -> dict[str, Any]:
    regressor_only = str(candidate_manifest.get("classifier_source", "")) == "none"
    metrics_run_id = (
        str(candidate_manifest["regressor_run_id"])
        if regressor_only
        else str(candidate_manifest["complete_run_id"])
    )
    run = client.get_run(metrics_run_id)
    metrics = cast(dict[str, float], run.data.metrics)
    elite_cfg = cast(dict[str, Any], candidate_manifest.get("elite_config", {}))

    row: dict[str, Any] = {
        "candidate_id": candidate_manifest["candidate_id"],
        "label": candidate_manifest["label"],
        "classifier_source": candidate_manifest["classifier_source"],
        "classifier_run_id": candidate_manifest["classifier_run_id"],
        "regressor_run_id": candidate_manifest["regressor_run_id"],
        "complete_run_id": candidate_manifest["complete_run_id"],
        "classifier_target_col": candidate_manifest["classifier_target_col"],
        "regressor_target_col": candidate_manifest["regressor_target_col"],
        "regressor_target_space": candidate_manifest["regressor_target_space"],
        "regressor_model": candidate_manifest["regressor_model"],
        "regressor_loss_function": candidate_manifest["regressor_loss_function"],
        "adjustment_method": candidate_manifest.get("adjustment_method"),
        "ablation_signature": candidate_manifest.get("ablation_signature", ""),
        "elite_source_column": elite_cfg.get("source_column"),
        "elite_quantile": elite_cfg.get("quantile"),
        "elite_scope": elite_cfg.get("scope"),
        "elite_fallback_scope": elite_cfg.get("fallback_scope"),
    }
    metric_columns = REGRESSOR_METRIC_COLUMNS if regressor_only else CORE_METRIC_COLUMNS
    for metric_name in metric_columns:
        row[metric_name] = float(metrics.get(metric_name, 0.0))
    return row


def export_matrix_report(cfg: DictConfig, *, exp_root: str | None = None) -> dict[str, Any]:
    exp_root = exp_root or resolve_exp_root(
        os.path.join(os.getcwd(), "pipelines", "export_matrix_report.py")
    )
    manifest_path = resolve_matrix_manifest_path(cfg, exp_root=exp_root)
    manifest = load_matrix_manifest(manifest_path)

    logger = ExperimentLogger(exp_root, f"Matrix_Report_{cfg.matrix.matrix_name}")
    logger.init_tracking()
    if logger.client is None:
        raise RuntimeError("MLflow tracking client was not initialized.")

    rows = [
        flatten_candidate_summary(logger.client, candidate)
        for candidate in load_candidate_manifests(manifest)
    ]
    table = pl.DataFrame(rows)

    output_path = cfg.get("output_path")
    if output_path:
        resolved_output_path = os.path.abspath(str(output_path))
    else:
        resolved_output_path = os.path.join(os.path.dirname(manifest_path), "candidate_summary.csv")
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
    table.write_csv(resolved_output_path)
    return {
        "manifest_path": manifest_path,
        "output_path": resolved_output_path,
        "row_count": len(rows),
    }
