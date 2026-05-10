import logging
import warnings

import hydra
from fof8_ml.orchestration.experiment_matrix import run_experiment_matrix
from fof8_ml.orchestration.pipeline_runner import resolve_exp_root
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="conf", config_name="experiment_matrix_pipeline")
def main(cfg: DictConfig) -> int:
    result = run_experiment_matrix(cfg, exp_root=resolve_exp_root(__file__))
    print(
        "Experiment matrix complete: "
        f"matrix={result['matrix_name']} candidates={result['candidate_count']} "
        f"manifest={result['matrix_manifest_path']}"
    )
    return int(result["candidate_count"])


if __name__ == "__main__":
    main()
