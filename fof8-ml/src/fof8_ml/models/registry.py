"""Explicit model registry for role-aware wrapper resolution.

This module defines the canonical mapping from `(role, model_key)` to
wrapper family and constructor. New model additions should be represented as:
1) one model config file under `pipelines/conf/model/`
2) one registry entry here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from fof8_ml.models.base import ModelFamily, ModelRole, ModelWrapper
from fof8_ml.models.catboost_wrapper import CatBoostClassifierWrapper, CatBoostRegressorWrapper
from fof8_ml.models.sklearn_wrapper import SklearnMLPRegressorWrapper, SklearnRegressorWrapper
from fof8_ml.models.xgboost_wrapper import XGBoostClassifierWrapper, XGBoostRegressorWrapper

ModelKey = str


@dataclass(frozen=True)
class ModelRegistration:
    """Immutable metadata for one registered model key."""

    role: ModelRole
    key: ModelKey
    family: ModelFamily
    builder: Callable[..., ModelWrapper]


_REGISTRY: dict[tuple[ModelRole, ModelKey], ModelRegistration] = {}


def _normalize_model_key(name: str) -> str:
    """Normalize model names for case-insensitive lookup."""

    return name.strip().lower()


def _normalize_role(role: ModelRole | str) -> ModelRole:
    normalized = role.strip().lower()
    if normalized not in {"classifier", "regressor"}:
        valid_roles = ", ".join(sorted({"classifier", "regressor"}))
        raise ValueError(f"Unknown role '{role}'. Valid roles: {valid_roles}")
    return cast(ModelRole, normalized)


def register_model(
    role: ModelRole,
    key: str,
    family: ModelFamily,
    builder: Callable[..., ModelWrapper],
) -> None:
    """Register a model wrapper builder for a `(role, model_key)` pair.

    Raises:
        ValueError: If the normalized role/key pair is already registered.
    """

    role_key = _normalize_role(role)
    model_key = _normalize_model_key(key)
    registry_key = (role_key, model_key)

    if registry_key in _REGISTRY:
        raise ValueError(f"Duplicate model registration for role='{role_key}', key='{model_key}'")

    _REGISTRY[registry_key] = ModelRegistration(
        role=role_key,
        key=model_key,
        family=family,
        builder=builder,
    )


def resolve_model(role: str, model_name: str) -> ModelRegistration:
    """Resolve a model registration for a role and model name.

    Raises:
        ValueError: If role is unknown or if the model key is not registered
            for that role. Error messages include valid options.
    """

    role_key = _normalize_role(role)
    model_key = _normalize_model_key(model_name)

    registration = _REGISTRY.get((role_key, model_key))
    if registration is None:
        valid_keys = ", ".join(list_model_keys(role_key))
        raise ValueError(
            f"Unknown model '{model_name}' for role '{role}'. "
            f"Valid model keys: {valid_keys}. "
            "Register new models in fof8_ml.models.registry."
        )
    return registration


def list_model_keys(role: ModelRole | str) -> list[str]:
    """List sorted model keys registered for a role."""

    role_key = _normalize_role(role)
    keys = [key for (registered_role, key) in _REGISTRY if registered_role == role_key]
    return sorted(keys)


def get_model_family(role: ModelRole | str, model_name: str) -> ModelFamily | None:
    """Return wrapper family for a model key, or `None` if not registered."""

    registration = _REGISTRY.get((_normalize_role(role), _normalize_model_key(model_name)))
    return None if registration is None else registration.family


register_model("classifier", "catboost_classifier", "catboost", CatBoostClassifierWrapper)
register_model("classifier", "xgb_classifier", "xgb", XGBoostClassifierWrapper)

register_model("regressor", "catboost_tweedie_regressor", "catboost", CatBoostRegressorWrapper)
register_model("regressor", "catboost_regressor_tweedie", "catboost", CatBoostRegressorWrapper)
register_model("regressor", "catboost_regressor_rmse", "catboost", CatBoostRegressorWrapper)
register_model("regressor", "catboost_regressor_mae", "catboost", CatBoostRegressorWrapper)
register_model("regressor", "catboost_regressor_expectile", "catboost", CatBoostRegressorWrapper)
register_model("regressor", "xgb_regressor", "xgb", XGBoostRegressorWrapper)
register_model("regressor", "sklearn_tweedie_regressor", "sklearn", SklearnRegressorWrapper)
register_model("regressor", "sklearn_gamma_regressor", "sklearn", SklearnRegressorWrapper)
register_model("regressor", "sklearn_mlp_regressor", "sklearn", SklearnMLPRegressorWrapper)
