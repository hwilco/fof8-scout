import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from fof8_ml.orchestration.complete_model import (
    resolve_complete_model_inputs,
    run_complete_model_evaluation,
)
from fof8_ml.orchestration.data_loader import DataLoader
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from mlflow.entities import Run
from omegaconf import DictConfig, open_dict

logging.getLogger("matplotlib").setLevel(logging.ERROR)


@dataclass(frozen=True)
class CandidateRun:
    run_id: str
    proxy_score: float
    trial_num: str
    params: dict[str, str]


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson_correlation(
        _rank_average(np.asarray(x, dtype=float)).tolist(),
        _rank_average(np.asarray(y, dtype=float)).tolist(),
    )


def _compose_complete_cfg(
    *,
    exp_root: str,
    classifier_run_id: str,
    regressor_run_id: str,
    complete_experiment_name: str,
) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = os.path.join(exp_root, "pipelines", "conf")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="complete_model_pipeline",
            overrides=[
                f"classifier_run_id={classifier_run_id}",
                f"regressor_run_id={regressor_run_id}",
                f"complete_experiment_name={complete_experiment_name}",
            ],
        )
    return cfg


def _metric(run: Run, name: str) -> float | None:
    value = run.data.metrics.get(name)
    return float(value) if value is not None else None


def _select_candidates(
    runs: list[Run],
    *,
    proxy_metric: str,
    top_n: int,
    maximize: bool,
    model_name: str | None,
) -> list[CandidateRun]:
    candidates: list[CandidateRun] = []
    for run in runs:
        score = _metric(run, proxy_metric)
        if score is None:
            continue
        if run.data.tags.get("model_role") != "regressor":
            continue
        if model_name is not None and run.data.params.get("model.name") != model_name:
            continue
        candidates.append(
            CandidateRun(
                run_id=run.info.run_id,
                proxy_score=score,
                trial_num=run.data.tags.get("trial_num", ""),
                params={
                    key: value
                    for key, value in run.data.params.items()
                    if key.startswith("model.params.")
                    or key.startswith("regressor.")
                    or key.startswith("ablation.toggles.")
                },
            )
        )

    return sorted(candidates, key=lambda item: item.proxy_score, reverse=maximize)[:top_n]


def _write_outputs(
    *,
    exp_root: str,
    output_prefix: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path]:
    output_dir = Path(exp_root) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    json_path = output_dir / f"{output_prefix}.json"
    pl.DataFrame(rows).write_csv(csv_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate top regressor sweep trials through the fixed-classifier complete model "
            "and measure proxy-vs-downstream metric correlation."
        )
    )
    parser.add_argument(
        "--source-experiment-name",
        default="Regressor_Sklearn_MLP_Top3_Hyperopt_v1",
        help="MLflow experiment containing the regressor sweep runs.",
    )
    parser.add_argument(
        "--complete-experiment-name",
        default="Complete_Model_Sweep_Proxy_Correlation",
        help="MLflow experiment where complete-model evaluation runs are logged.",
    )
    parser.add_argument("--classifier-run-id", required=True, help="Fixed classifier run id.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top regressor runs.")
    parser.add_argument(
        "--proxy-metric",
        default="regressor_val_draft_value_score",
        help="Regressor sweep metric used to rank candidate trials.",
    )
    parser.add_argument(
        "--downstream-metric",
        default="complete_draft_value_score",
        help="Complete-model metric used for correlation.",
    )
    parser.add_argument(
        "--minimize-proxy",
        action="store_true",
        help="Select lowest proxy metric values instead of highest.",
    )
    parser.add_argument(
        "--model-name",
        default="sklearn_mlp_regressor",
        help="Optional model.name filter for source regressor runs. Use empty string to disable.",
    )
    parser.add_argument(
        "--output-prefix",
        default="mlp_sweep_complete_correlation",
        help="Basename for CSV/JSON files written under outputs/.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=500,
        help="Maximum MLflow runs to scan from the source experiment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_root = resolve_exp_root(__file__)

    tracking_logger = ExperimentLogger(exp_root, args.complete_experiment_name)
    tracking_logger.init_tracking()
    if tracking_logger.client is None:
        raise RuntimeError("MLflow tracking client was not initialized.")
    client = tracking_logger.client

    source_exp = client.get_experiment_by_name(args.source_experiment_name)
    if source_exp is None:
        raise ValueError(f"MLflow experiment not found: {args.source_experiment_name}")

    order_direction = "ASC" if args.minimize_proxy else "DESC"
    runs = client.search_runs(
        experiment_ids=[source_exp.experiment_id],
        order_by=[f"metrics.{args.proxy_metric} {order_direction}"],
        max_results=args.max_runs,
    )
    model_name = args.model_name.strip() or None
    candidates = _select_candidates(
        runs,
        proxy_metric=args.proxy_metric,
        top_n=args.top_n,
        maximize=not args.minimize_proxy,
        model_name=model_name,
    )
    if not candidates:
        raise ValueError(
            f"No candidate regressor runs found in '{args.source_experiment_name}' with "
            f"metric '{args.proxy_metric}'."
        )

    cfg = _compose_complete_cfg(
        exp_root=exp_root,
        classifier_run_id=args.classifier_run_id,
        regressor_run_id=candidates[0].run_id,
        complete_experiment_name=args.complete_experiment_name,
    )
    with open_dict(cfg):
        cfg.quiet_sweep = True
    resolve_complete_model_inputs(cfg, exp_root=exp_root, client=client)

    data_loader = DataLoader(exp_root, quiet=True)
    data = data_loader.load(cfg)
    data = data_loader.apply_feature_ablation(
        data,
        cfg.get("include_features"),
        cfg.get("exclude_features"),
    )

    rows: list[dict[str, object]] = []
    downstream_scores: list[float] = []
    proxy_scores: list[float] = []

    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[{index}/{len(candidates)}] Evaluating regressor={candidate.run_id} "
            f"{args.proxy_metric}={candidate.proxy_score:.6f}"
        )
        with open_dict(cfg):
            cfg.regressor_run_id = candidate.run_id

        inputs = resolve_complete_model_inputs(
            cfg,
            exp_root=exp_root,
            client=client,
        )
        with tracking_logger.start_pipeline_run(
            f"CompleteProxyCheck_{candidate.run_id[:8]}",
            tags={
                "evaluation_type": "complete_model_proxy_correlation",
                "source_regressor_experiment": args.source_experiment_name,
                "source_regressor_run_id": candidate.run_id,
            },
        ):
            metrics = run_complete_model_evaluation(
                cfg=cfg,
                data=data,
                exp_root=exp_root,
                logger=tracking_logger,
                inputs=inputs,
            )
            mlflow.log_metric(args.proxy_metric, candidate.proxy_score)

        downstream_score = float(metrics[args.downstream_metric])
        proxy_scores.append(candidate.proxy_score)
        downstream_scores.append(downstream_score)
        row: dict[str, object] = {
            "rank_by_proxy": index,
            "regressor_run_id": candidate.run_id,
            "trial_num": candidate.trial_num,
            args.proxy_metric: candidate.proxy_score,
            args.downstream_metric: downstream_score,
        }
        row.update(candidate.params)
        rows.append(row)

    best_by_proxy = rows[0]
    best_by_downstream = max(rows, key=lambda row: float(str(row[args.downstream_metric])))
    summary: dict[str, object] = {
        "source_experiment_name": args.source_experiment_name,
        "complete_experiment_name": args.complete_experiment_name,
        "classifier_run_id": args.classifier_run_id,
        "top_n": len(candidates),
        "proxy_metric": args.proxy_metric,
        "downstream_metric": args.downstream_metric,
        "pearson": pearson_correlation(proxy_scores, downstream_scores),
        "spearman": spearman_correlation(proxy_scores, downstream_scores),
        "best_by_proxy": best_by_proxy,
        "best_by_downstream": best_by_downstream,
    }
    csv_path, json_path = _write_outputs(
        exp_root=exp_root,
        output_prefix=args.output_prefix,
        rows=rows,
        summary=summary,
    )

    print("\nCorrelation summary")
    print(f"Pearson:  {summary['pearson']}")
    print(f"Spearman: {summary['spearman']}")
    print(f"Best by proxy:      {best_by_proxy['regressor_run_id']}")
    print(f"Best by downstream: {best_by_downstream['regressor_run_id']}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
