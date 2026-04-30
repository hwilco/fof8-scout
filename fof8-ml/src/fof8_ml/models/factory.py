from fof8_ml.models.base import ModelWrapper
from fof8_ml.models.catboost_wrapper import CatBoostClassifierWrapper, CatBoostRegressorWrapper
from fof8_ml.models.sklearn_wrapper import SklearnRegressorWrapper
from fof8_ml.models.xgboost_wrapper import XGBoostClassifierWrapper, XGBoostRegressorWrapper


def get_model_wrapper(
    model_name: str,
    stage: str,
    random_seed: int,
    params: dict,
    use_gpu: bool = False,
    thread_count: int = -1,
) -> ModelWrapper:
    """
    Factory function to instantiate the appropriate model wrapper based on config.

    Args:
        model_name: Name of the model (e.g., 's1_catboost', 'xgb').
        stage: Pipeline stage ('stage1' or 'stage2').
        random_seed: Random seed for reproducibility.
        params: Dictionary of model hyperparameters.
        use_gpu: Whether to enable GPU acceleration.
        thread_count: Number of threads to use (-1 for all cores).

    Returns:
        An instantiated subclass of ModelWrapper.
    """
    model_name = model_name.lower()
    if stage == "stage1":
        if "catboost" in model_name:
            return CatBoostClassifierWrapper(
                random_seed=random_seed, use_gpu=use_gpu, thread_count=thread_count, **params
            )
        elif "xgb" in model_name:
            return XGBoostClassifierWrapper(random_seed=random_seed, use_gpu=use_gpu, **params)
        else:
            raise ValueError(f"Unknown model for stage 1: {model_name}")
    elif stage == "stage2":
        if "catboost" in model_name:
            return CatBoostRegressorWrapper(
                random_seed=random_seed, use_gpu=use_gpu, thread_count=thread_count, **params
            )
        elif "xgb" in model_name:
            return XGBoostRegressorWrapper(random_seed=random_seed, use_gpu=use_gpu, **params)
        elif "sklearn" in model_name or "tweedie" in model_name or "gamma" in model_name:
            return SklearnRegressorWrapper(model_name=model_name, use_gpu=use_gpu, **params)
        else:
            raise ValueError(f"Unknown model for stage 2: {model_name}")


def apply_quiet_params(model_name: str, params: dict) -> dict:
    """Modifies params dictionary to silence output for sweeps."""
    quiet_params = params.copy()
    name_lower = model_name.lower()
    if "catboost" in name_lower:
        quiet_params["logging_level"] = "Silent"
    elif "xgb" in name_lower:
        quiet_params["verbosity"] = 0
    return quiet_params
