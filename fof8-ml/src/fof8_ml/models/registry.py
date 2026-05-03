"""Explicit model registry for stage-aware wrapper resolution.

This module defines the canonical mapping from `(stage, model_key)` to
wrapper family and constructor. New model additions should be represented as:
1) one model config file under `pipelines/conf/model/`
2) one registry entry here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fof8_ml.models.base import ModelWrapper
from fof8_ml.models.catboost_wrapper import CatBoostClassifierWrapper, CatBoostRegressorWrapper
from fof8_ml.models.sklearn_wrapper import SklearnRegressorWrapper
from fof8_ml.models.xgboost_wrapper import XGBoostClassifierWrapper, XGBoostRegressorWrapper

Stage = str
ModelKey = str


@dataclass(frozen=True)
class ModelRegistration:
    """Immutable metadata for one registered model key."""

    stage: Stage
    key: ModelKey
    family: str
    builder: Callable[..., ModelWrapper]


_REGISTRY: dict[tuple[Stage, ModelKey], ModelRegistration] = {}


def _normalize(name: str) -> str:
    """Normalize stage and model names for case-insensitive lookup."""

    return name.strip().lower()


def register_model(stage: str, key: str, family: str, builder: Callable[..., ModelWrapper]) -> None:
    """Register a model wrapper builder for a `(stage, model_key)` pair.

    Raises:
        ValueError: If the normalized stage/key pair is already registered.
    """

    stage_key = _normalize(stage)
    model_key = _normalize(key)
    registry_key = (stage_key, model_key)

    if registry_key in _REGISTRY:
        raise ValueError(f"Duplicate model registration for stage='{stage_key}', key='{model_key}'")

    _REGISTRY[registry_key] = ModelRegistration(
        stage=stage_key,
        key=model_key,
        family=family,
        builder=builder,
    )


def resolve_model(stage: str, model_name: str) -> ModelRegistration:
    """Resolve a model registration for a stage and model name.

    Raises:
        ValueError: If stage is unknown or if the model key is not registered
            for that stage. Error messages include valid options.
    """

    stage_key = _normalize(stage)
    model_key = _normalize(model_name)
    registered_stages = {registered_stage for (registered_stage, _) in _REGISTRY}
    if stage_key not in registered_stages:
        valid_stages = ", ".join(sorted(registered_stages))
        raise ValueError(f"Unknown stage '{stage}'. Valid stages: {valid_stages}")

    registration = _REGISTRY.get((stage_key, model_key))
    if registration is None:
        valid_keys = ", ".join(list_model_keys(stage_key))
        raise ValueError(
            f"Unknown model '{model_name}' for stage '{stage}'. "
            f"Valid model keys: {valid_keys}. "
            "Register new models in fof8_ml.models.registry."
        )
    return registration


def list_model_keys(stage: str) -> list[str]:
    """List sorted model keys registered for a stage."""

    stage_key = _normalize(stage)
    keys = [key for (registered_stage, key) in _REGISTRY if registered_stage == stage_key]
    return sorted(keys)


def get_model_family(stage: str, model_name: str) -> str | None:
    """Return wrapper family for a model key, or `None` if not registered."""

    registration = _REGISTRY.get((_normalize(stage), _normalize(model_name)))
    return None if registration is None else registration.family


# Stage 1 classifier model aliases.
register_model("stage1", "s1_catboost", "catboost", CatBoostClassifierWrapper)
register_model("stage1", "catboost_career_threshold", "catboost", CatBoostClassifierWrapper)
register_model("stage1", "career_threshold_catboost", "catboost", CatBoostClassifierWrapper)

register_model("stage1", "xgb", "xgb", XGBoostClassifierWrapper)
register_model("stage1", "career_threshold_xgb", "xgb", XGBoostClassifierWrapper)
register_model("stage1", "xgboost_career_threshold", "xgb", XGBoostClassifierWrapper)

# Stage 2 regressor model aliases.
register_model("stage2", "s2_catboost", "catboost", CatBoostRegressorWrapper)
register_model("stage2", "catboost_career_threshold", "catboost", CatBoostRegressorWrapper)
register_model("stage2", "career_threshold_catboost", "catboost", CatBoostRegressorWrapper)

register_model("stage2", "s2_xgb", "xgb", XGBoostRegressorWrapper)
register_model("stage2", "xgb", "xgb", XGBoostRegressorWrapper)
register_model("stage2", "career_threshold_xgb", "xgb", XGBoostRegressorWrapper)
register_model("stage2", "xgboost_career_threshold", "xgb", XGBoostRegressorWrapper)

register_model("stage2", "reg_tweedie", "sklearn", SklearnRegressorWrapper)
register_model("stage2", "reg_gamma", "sklearn", SklearnRegressorWrapper)
register_model("stage2", "sklearn_tweedie", "sklearn", SklearnRegressorWrapper)
register_model("stage2", "sklearn_gamma", "sklearn", SklearnRegressorWrapper)
