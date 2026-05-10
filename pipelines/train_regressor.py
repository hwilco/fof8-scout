import logging
import warnings

import hydra
import matplotlib
from fof8_ml.orchestration.pipeline_runner import (
    build_pipeline_context,
    finalize_pipeline_run,
    select_optimization_metric,
)
from fof8_ml.orchestration.regressor import run_regressor
from omegaconf import DictConfig

# Suppress Optuna deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
# Suppress Hydra experimental feature warnings
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="conf", config_name="regressor_pipeline")
def main(cfg: DictConfig) -> float:
    """Train and evaluate the Regressor Intensity Regressor.

    Training is performed on ground truth positive cases (y_cls == 1) only.

    Args:
        cfg: Hydra config from regressor_pipeline.yaml.

    Returns:
        The optimization metric score for the current trial.
    """
    ctx = build_pipeline_context(cfg, __file__)

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

        available_metrics = run_regressor(ctx)
        ctx.logger.write_dvc_json(
            {
                "run_id": pipeline_run.info.run_id,
                "model_role": "regressor",
                "optimization_metric": cfg.optimization.metric,
                "optimization_score": float(available_metrics[cfg.optimization.metric]),
            },
            "regressor_run.json",
        )

        if not ctx.sweep_context.quiet:
            print("\nRegressor Training Complete. Model saved to MLflow.")

        opt_metric = cfg.optimization.metric
        current_score = select_optimization_metric(available_metrics, opt_metric)

        return finalize_pipeline_run(
            ctx=ctx,
            pipeline_run_id=pipeline_run.info.run_id,
            metric_name=opt_metric,
            current_score=current_score,
            metrics_filename="regressor_metrics.json",
        )


if __name__ == "__main__":
    main()
