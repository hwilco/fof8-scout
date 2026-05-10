import logging
import warnings

import hydra
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root
from fof8_ml.reporting.phase4_diagnostics import export_phase4_diagnostics
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="conf", config_name="phase4_diagnostics_pipeline")
def main(cfg: DictConfig) -> str:
    result = export_phase4_diagnostics(cfg, exp_root=resolve_exp_root(__file__))
    print(
        "Phase 4 diagnostics exported: "
        f"candidates={result['candidate_count']} output_dir={result['output_dir']}"
    )
    return str(result["output_dir"])


if __name__ == "__main__":
    main()
