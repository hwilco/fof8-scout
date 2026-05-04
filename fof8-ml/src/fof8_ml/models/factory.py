"""Model factory backed by the explicit role/model registry."""

from typing import Any

from fof8_ml.models.base import ModelWrapper
from fof8_ml.models.registry import get_model_family, resolve_model


def get_model_wrapper(
    model_name: str,
    role: str,
    random_seed: int,
    params: dict[str, Any],
    use_gpu: bool = False,
    thread_count: int = -1,
) -> ModelWrapper:
    """
    Instantiate a role-specific model wrapper from a registered model key.

    Model names are resolved exclusively via `fof8_ml.models.registry`.
    Any model used by config must be registered there first.

    Args:
        model_name: Name/alias of the model from config.
        role: Model role ('classifier' or 'regressor').
        random_seed: Random seed for reproducibility.
        params: Dictionary of model hyperparameters.
        use_gpu: Whether to enable GPU acceleration.
        thread_count: Number of threads to use (-1 for all cores).

    Returns:
        An instantiated subclass of ModelWrapper.
    """
    registration = resolve_model(role=role, model_name=model_name)

    if registration.family == "catboost":
        return registration.builder(
            random_seed=random_seed,
            use_gpu=use_gpu,
            thread_count=thread_count,
            **params,
        )
    if registration.family == "xgb":
        return registration.builder(random_seed=random_seed, use_gpu=use_gpu, **params)
    if registration.family == "sklearn":
        return registration.builder(model_name=model_name.lower(), use_gpu=use_gpu, **params)

    raise ValueError(
        "Unsupported model family "
        f"'{registration.family}' for role '{role}' and model '{model_name}'"
    )


def apply_quiet_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return params with quiet flags applied for CatBoost/XGBoost models.

    The model family is resolved from the explicit registry across known
    roles. Non-CatBoost/XGBoost models are returned unchanged.
    """

    quiet_params = params.copy()

    for role in ("classifier", "regressor"):
        family = get_model_family(role=role, model_name=model_name)
        if family == "catboost":
            quiet_params["logging_level"] = "Silent"
            return quiet_params
        if family == "xgb":
            quiet_params["verbosity"] = 0
            return quiet_params

    return quiet_params
