import json
import logging
import os
import warnings

import hydra
import mlflow
import polars as pl
from fof8_ml.evaluation.complete_model import (
    evaluate_complete_model,
    load_complete_model,
    predict_complete_model,
)
from fof8_ml.orchestration.data_loader import DataLoader
from fof8_ml.orchestration.evaluator import fit_elite_thresholds
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root
from omegaconf import DictConfig, open_dict

warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


def _scope_meta_from_features(X: pl.DataFrame, elite_cfg: DictConfig | None) -> pl.DataFrame | None:
    scope_column = "Position_Group"
    if elite_cfg is not None:
        scope_column = str(elite_cfg.get("scope_column", scope_column))
    if scope_column in X.columns:
        return X.select(pl.col(scope_column).cast(pl.Utf8))
    return None


def _resolve_run_target_param(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    param_name: str,
) -> str:
    run = client.get_run(run_id)
    value = run.data.params.get(param_name)
    if value is None:
        raise ValueError(f"Run '{run_id}' is missing required MLflow param '{param_name}'.")
    return str(value)


def _resolve_run_id_from_input(
    run_id: str | None,
    manifest_path: str | None,
    *,
    role: str,
    exp_root: str,
) -> str:
    if run_id:
        return str(run_id)
    if not manifest_path:
        raise ValueError(f"Provide either {role}_run_id=<run_id> or {role}_run_manifest=<path>.")
    resolved_manifest_path = (
        manifest_path if os.path.isabs(manifest_path) else os.path.join(exp_root, manifest_path)
    )
    with open(resolved_manifest_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    resolved_run_id = payload.get("run_id")
    if not resolved_run_id:
        raise ValueError(f"Manifest '{resolved_manifest_path}' is missing required key 'run_id'.")
    return str(resolved_run_id)


@hydra.main(version_base=None, config_path="conf", config_name="regressor_pipeline")
def main(cfg: DictConfig) -> float:
    exp_root = resolve_exp_root(__file__)
    classifier_run_id = _resolve_run_id_from_input(
        cfg.get("classifier_run_id"),
        cfg.get("classifier_run_manifest"),
        role="classifier",
        exp_root=exp_root,
    )
    regressor_run_id = _resolve_run_id_from_input(
        cfg.get("regressor_run_id"),
        cfg.get("regressor_run_manifest"),
        role="regressor",
        exp_root=exp_root,
    )
    logger = ExperimentLogger(
        exp_root,
        str(cfg.get("complete_experiment_name", "Complete_Model_Evaluation")),
    )
    logger.init_tracking()
    if logger.client is None:
        raise RuntimeError("MLflow tracking client was not initialized.")

    classifier_target_col = _resolve_run_target_param(
        logger.client,
        str(classifier_run_id),
        "target.classifier.target_col",
    )
    regressor_target_col = _resolve_run_target_param(
        logger.client,
        str(regressor_run_id),
        "target.regressor.target_col",
    )

    with open_dict(cfg):
        cfg.target.classifier_sieve.target_col = classifier_target_col
        cfg.target.regressor_intensity.target_col = regressor_target_col

    data_loader = DataLoader(exp_root, quiet=False)
    data = data_loader.load(cfg)
    data_loader.print_summary(data, cfg)

    complete_bundle = load_complete_model(
        logger.client,
        classifier_run_id=str(classifier_run_id),
        regressor_run_id=str(regressor_run_id),
    )

    X_eval = data.X_test
    if len(X_eval) == 0:
        raise ValueError(
            "Held-out evaluation split is empty; complete model evaluation cannot run."
        )
    eval_group = (
        data.meta_test.get_column("Universe").cast(pl.Utf8)
        + ":"
        + data.meta_test.get_column("Year").cast(pl.Utf8)
    ).to_numpy()
    elite_cfg = cfg.target.get("outcome_scorecard", {}).get("elite")
    elite_thresholds = fit_elite_thresholds(
        data.outcomes_train,
        _scope_meta_from_features(data.X_train, elite_cfg),
        elite_cfg,
    )

    prediction_dict = predict_complete_model(
        X_eval,
        complete_bundle.classifier,
        complete_bundle.regressor,
    )

    metrics = evaluate_complete_model(
        y_true=data.y_reg_test,
        y_pred=prediction_dict["complete_prediction"],
        draft_year=eval_group,
        outcome_columns=data.outcomes_test,
        meta_columns=_scope_meta_from_features(X_eval, elite_cfg),
        elite_cfg=elite_cfg,
        elite_thresholds=elite_thresholds,
    )

    board_df = data.meta_test.with_columns(
        [
            pl.Series(
                "Position_Group",
                X_eval.get_column("Position_Group").cast(pl.Utf8).to_list()
                if "Position_Group" in X_eval.columns
                else ["unknown"] * len(X_eval),
            ),
            pl.Series("classifier_probability", prediction_dict["classifier_probability"]),
            pl.Series("regressor_prediction", prediction_dict["regressor_prediction"]),
            pl.Series("complete_prediction", prediction_dict["complete_prediction"]),
            pl.Series("actual_target", data.y_reg_test),
        ]
    )
    board_df = board_df.with_columns(
        pl.col("complete_prediction")
        .rank(method="ordinal", descending=True)
        .over(["Universe", "Year"])
        .alias("rank_within_year")
    ).sort(["Universe", "Year", "rank_within_year"])

    artifact_name = os.path.join(exp_root, "outputs", "complete_model_holdout_board.csv")
    os.makedirs(os.path.dirname(artifact_name), exist_ok=True)
    board_df.write_csv(artifact_name)

    with logger.start_pipeline_run(
        f"CompleteModel_{classifier_run_id[:8]}_{regressor_run_id[:8]}",
        tags={"evaluation_type": "complete_model"},
    ) as run:
        logger.log_data_summary(
            data,
            cfg,
            os.path.abspath(os.path.join(exp_root, cfg.data.raw_path)),
            False,
            None,
        )
        mlflow.log_params(
            {
                "classifier_run_id": classifier_run_id,
                "regressor_run_id": regressor_run_id,
                "complete_classifier_target_col": classifier_target_col,
                "complete_regressor_target_col": regressor_target_col,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(artifact_name)
        if elite_cfg is not None:
            mlflow.log_dict(
                {
                    "enabled": bool(elite_cfg.get("enabled", False)),
                    "source_column": elite_cfg.get("source_column"),
                    "quantile": elite_cfg.get("quantile"),
                    "scope": elite_cfg.get("scope"),
                    "scope_column": elite_cfg.get("scope_column"),
                    "fallback_scope": elite_cfg.get("fallback_scope"),
                    "min_group_size": elite_cfg.get("min_group_size"),
                    "top_k_precision": elite_cfg.get("top_k_precision"),
                    "top_k_recall": elite_cfg.get("top_k_recall"),
                    "thresholds": elite_thresholds or {},
                },
                "complete_model_elite_config.json",
            )
        score = float(metrics["complete_draft_value_score"])
        logger.write_dvc_metrics(
            "complete_draft_value_score",
            score,
            metrics_filename="complete_model_metrics.json",
        )
        logger.write_dvc_json(
            {
                "run_id": run.info.run_id,
                "evaluation_type": "complete_model",
                "classifier_run_id": classifier_run_id,
                "regressor_run_id": regressor_run_id,
                "optimization_metric": "complete_draft_value_score",
                "optimization_score": score,
            },
            "complete_model_run.json",
        )
        return score


if __name__ == "__main__":
    main()
