import logging
import warnings

import hydra
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root
from fof8_ml.reporting.phase4_report import export_phase4_report
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="conf", config_name="phase4_report_pipeline")
def main(cfg: DictConfig) -> str:
    result = export_phase4_report(cfg, exp_root=resolve_exp_root(__file__))
    print(f"Phase 4 report exported: rows={result['row_count']} output={result['output_path']}")
    return str(result["output_path"])


if __name__ == "__main__":
    main()
