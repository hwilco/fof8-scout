import logging
import os
import warnings

import hydra
from fof8_ml.orchestration.complete_model import (
    resolve_complete_model_inputs,
    run_complete_model_evaluation,
)
from fof8_ml.orchestration.data_loader import DataLoader
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root, select_optimization_metric
from fof8_ml.orchestration.sweep_manager import SweepManager
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="conf", config_name="complete_model_pipeline")
def main(cfg: DictConfig) -> float:
    """Run complete-model evaluation pipeline and return optimization score.

    Args:
        cfg: Runtime Hydra configuration.

    Returns:
        Optimization metric value for sweep orchestration.
    """

    exp_root = resolve_exp_root(__file__)
    absolute_raw_path = os.path.abspath(os.path.join(exp_root, cfg.data.raw_path))

    logger = ExperimentLogger(
        exp_root,
        str(cfg.get("complete_experiment_name", "Complete_Model_Evaluation")),
    )
    logger.init_tracking()
    if logger.client is None or logger.experiment_id is None:
        raise RuntimeError("MLflow tracking client/experiment were not initialized.")

    inputs = resolve_complete_model_inputs(
        cfg,
        exp_root=exp_root,
        client=logger.client,
    )

    sweep_mgr = SweepManager(logger.client, logger.experiment_id, exp_root)
    sweep_context = sweep_mgr.detect_sweep(cfg)

    data_loader = DataLoader(exp_root, quiet=sweep_context.quiet)
    data = data_loader.load(cfg)
    data_loader.print_summary(data, cfg)

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
        available_metrics = run_complete_model_evaluation(
            cfg=cfg,
            data=data,
            exp_root=exp_root,
            logger=logger,
            inputs=inputs,
        )
        opt_metric = "complete_draft_value_score"
        current_score = select_optimization_metric(available_metrics, opt_metric)

        if sweep_context.is_sweep:
            is_new_best = sweep_mgr.update_champion(
                sweep_context, pipeline_run.info.run_id, current_score, cfg
            )
            sweep_mgr.print_leaderboard(sweep_context, current_score, is_new_best, cfg)

        logger.write_dvc_metrics(
            opt_metric,
            current_score,
            metrics_filename="complete_model_metrics.json",
        )
        return float(current_score)


if __name__ == "__main__":
    main()
