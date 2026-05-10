from __future__ import annotations

import itertools
import os
from typing import Any

import polars as pl
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from fof8_ml.evaluation.complete_model import evaluate_complete_model_by_slice
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root
from fof8_ml.reporting.matrix_report import (
    load_candidate_manifests,
    load_matrix_manifest,
    resolve_matrix_manifest_path,
)


def _download_csv(client: MlflowClient, run_id: str, artifact_path: str) -> pl.DataFrame:
    local_path = client.download_artifacts(run_id, artifact_path)
    return pl.read_csv(local_path)


def _artifact_exists(client: MlflowClient, run_id: str, artifact_path: str) -> bool:
    parent = os.path.dirname(artifact_path)
    target = artifact_path
    artifacts = client.list_artifacts(run_id, parent)
    return any(getattr(artifact, "path", "") == target for artifact in artifacts)


def _fallback_position_metrics_from_board(board: pl.DataFrame) -> pl.DataFrame:
    draft_group = (
        board.get_column("Universe").cast(pl.Utf8) + ":" + board.get_column("Year").cast(pl.Utf8)
    ).to_numpy()
    return evaluate_complete_model_by_slice(
        y_true=board.get_column("actual_target").to_numpy(),
        y_pred=board.get_column("complete_prediction").to_numpy(),
        draft_year=draft_group,
        slice_values=board.get_column("Position_Group").cast(pl.Utf8).to_numpy(),
        slice_column="Position_Group",
        outcome_columns=None,
        meta_columns=board.select(pl.col("Position_Group").cast(pl.Utf8)),
    )


def _topk_overlap_rows(
    candidate_a: dict[str, Any], candidate_b: dict[str, Any], k: int
) -> dict[str, Any]:
    board_a = candidate_a["board"]
    board_b = candidate_b["board"]
    keys = ["Universe", "Year", "Player_ID"]
    top_a = board_a.filter(pl.col("rank_within_year") <= k).select(keys)
    top_b = board_b.filter(pl.col("rank_within_year") <= k).select(keys)
    overlap_count = top_a.join(top_b, on=keys, how="inner").height
    denom = max(top_a.height, top_b.height, 1)
    return {
        "candidate_a": candidate_a["candidate_id"],
        "candidate_b": candidate_b["candidate_id"],
        "k": k,
        "top_a_count": top_a.height,
        "top_b_count": top_b.height,
        "overlap_count": overlap_count,
        "overlap_rate": float(overlap_count / denom),
    }


def _rank_delta_rows(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> pl.DataFrame:
    join_keys = ["Universe", "Year", "Player_ID", "First_Name", "Last_Name", "Position_Group"]
    merged = (
        candidate_a["board"]
        .select(
            join_keys
            + [
                pl.col("rank_within_year").alias("rank_a"),
                pl.col("complete_prediction").alias("score_a"),
            ]
        )
        .join(
            candidate_b["board"].select(
                join_keys
                + [
                    pl.col("rank_within_year").alias("rank_b"),
                    pl.col("complete_prediction").alias("score_b"),
                ]
            ),
            on=join_keys,
            how="inner",
        )
        .with_columns(
            [
                (pl.col("rank_b") - pl.col("rank_a")).alias("rank_delta"),
                (pl.col("rank_b") - pl.col("rank_a")).abs().alias("abs_rank_delta"),
                pl.lit(candidate_a["candidate_id"]).alias("candidate_a"),
                pl.lit(candidate_b["candidate_id"]).alias("candidate_b"),
            ]
        )
        .sort(["abs_rank_delta", "Universe", "Year"], descending=[True, False, False])
    )
    return merged


def _position_mix_rows(candidate: dict[str, Any], k: int) -> pl.DataFrame:
    top_board = candidate["board"].filter(pl.col("rank_within_year") <= k)
    total = max(top_board.height, 1)
    return (
        top_board.group_by("Position_Group")
        .len()
        .rename({"len": "count"})
        .with_columns(
            [
                pl.lit(candidate["candidate_id"]).alias("candidate_id"),
                pl.lit(candidate["label"]).alias("label"),
                pl.lit(k).alias("k"),
                (pl.col("count") / total).alias("share"),
            ]
        )
        .sort(["candidate_id", "k", "count"], descending=[False, False, True])
    )


def export_matrix_diagnostics(cfg: DictConfig, *, exp_root: str | None = None) -> dict[str, Any]:
    exp_root = exp_root or resolve_exp_root(
        os.path.join(os.getcwd(), "pipelines", "export_matrix_diagnostics.py")
    )
    manifest_path = resolve_matrix_manifest_path(cfg, exp_root=exp_root)
    matrix_manifest = load_matrix_manifest(manifest_path)
    candidate_manifests = load_candidate_manifests(matrix_manifest)

    logger = ExperimentLogger(exp_root, f"Matrix_Diagnostics_{cfg.matrix.matrix_name}")
    logger.init_tracking()
    if logger.client is None:
        raise RuntimeError("MLflow tracking client was not initialized.")

    enriched_candidates: list[dict[str, Any]] = []
    for candidate in candidate_manifests:
        enriched = dict(candidate)
        enriched["board"] = _download_csv(
            logger.client,
            candidate["complete_run_id"],
            str(candidate.get("board_artifact_path", "complete_model_holdout_board.csv")),
        )
        position_artifact_path = str(
            candidate.get(
                "position_group_metrics_artifact_path",
                "complete_model_position_group_metrics.csv",
            )
        )
        try:
            if not _artifact_exists(
                logger.client, candidate["complete_run_id"], position_artifact_path
            ):
                raise FileNotFoundError(position_artifact_path)
            enriched["position_metrics"] = _download_csv(
                logger.client,
                candidate["complete_run_id"],
                position_artifact_path,
            )
        except Exception:
            enriched["position_metrics"] = _fallback_position_metrics_from_board(enriched["board"])
        enriched_candidates.append(enriched)

    output_dir = cfg.get("output_dir")
    resolved_output_dir = (
        os.path.abspath(str(output_dir)) if output_dir else os.path.dirname(manifest_path)
    )
    os.makedirs(resolved_output_dir, exist_ok=True)

    position_rows = []
    for candidate in enriched_candidates:
        position_rows.append(
            candidate["position_metrics"].with_columns(
                [
                    pl.lit(candidate["candidate_id"]).alias("candidate_id"),
                    pl.lit(candidate["label"]).alias("label"),
                ]
            )
        )
    position_summary = (
        pl.concat(position_rows, how="vertical_relaxed") if position_rows else pl.DataFrame()
    )
    position_summary_path = os.path.join(resolved_output_dir, "position_group_summary.csv")
    position_summary.write_csv(position_summary_path)

    overlap_rows = []
    rank_delta_frames = []
    for candidate_a, candidate_b in itertools.combinations(enriched_candidates, 2):
        overlap_rows.append(_topk_overlap_rows(candidate_a, candidate_b, k=32))
        overlap_rows.append(_topk_overlap_rows(candidate_a, candidate_b, k=64))
        rank_delta_frames.append(_rank_delta_rows(candidate_a, candidate_b))
    overlap_summary = pl.DataFrame(overlap_rows) if overlap_rows else pl.DataFrame()
    overlap_summary_path = os.path.join(resolved_output_dir, "board_overlap_summary.csv")
    overlap_summary.write_csv(overlap_summary_path)

    rank_deltas = (
        pl.concat(rank_delta_frames, how="vertical_relaxed")
        if rank_delta_frames
        else pl.DataFrame()
    )
    rank_deltas_path = os.path.join(resolved_output_dir, "board_rank_deltas.csv")
    rank_deltas.write_csv(rank_deltas_path)

    position_mix_frames = []
    for candidate in enriched_candidates:
        position_mix_frames.append(_position_mix_rows(candidate, k=32))
        position_mix_frames.append(_position_mix_rows(candidate, k=64))
    position_mix = (
        pl.concat(position_mix_frames, how="vertical_relaxed")
        if position_mix_frames
        else pl.DataFrame()
    )
    position_mix_path = os.path.join(resolved_output_dir, "board_position_mix.csv")
    position_mix.write_csv(position_mix_path)

    return {
        "manifest_path": manifest_path,
        "output_dir": resolved_output_dir,
        "position_summary_path": position_summary_path,
        "overlap_summary_path": overlap_summary_path,
        "rank_deltas_path": rank_deltas_path,
        "position_mix_path": position_mix_path,
        "candidate_count": len(enriched_candidates),
    }
