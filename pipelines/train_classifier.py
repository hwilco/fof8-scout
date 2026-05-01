import logging
import warnings

import hydra
import matplotlib
from fof8_ml.orchestration.pipeline_runner import (
    build_pipeline_context,
    finalize_pipeline_run,
    select_optimization_metric,
)
from fof8_ml.orchestration.stage1_classifier import run_classifier_stage
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


@hydra.main(version_base=None, config_path="conf", config_name="classifier_pipeline")
def main(cfg: DictConfig) -> float:
    """Train and evaluate the Stage 1 Sieve Classifier.

    Args:
        cfg: Hydra config from classifier_pipeline.yaml.

    Returns:
        The optimization metric score for the current trial.
    """
    ctx = build_pipeline_context(cfg, __file__)

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

        available_metrics = run_classifier_stage(ctx)

        if not ctx.sweep_context.quiet:
            print("\nClassifier Training Complete. Model saved to MLflow.")

        opt_metric = cfg.optimization.metric
        current_score = select_optimization_metric(available_metrics, opt_metric)

        return finalize_pipeline_run(
            ctx=ctx,
            pipeline_run_id=pipeline_run.info.run_id,
            metric_name=opt_metric,
            current_score=current_score,
            metrics_filename="classifier_metrics.json",
        )


if __name__ == "__main__":
    main()
