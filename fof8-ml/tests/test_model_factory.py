from pathlib import Path

import pytest
from fof8_ml.models import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    SklearnRegressorWrapper,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
)
from fof8_ml.models.factory import apply_quiet_params, get_model_wrapper
from fof8_ml.models.registry import list_model_keys, register_model, resolve_model
from omegaconf import OmegaConf

MODEL_CFG_DIR = Path(__file__).resolve().parents[2] / "pipelines" / "conf" / "model"


@pytest.mark.parametrize(
    ("model_name", "role", "expected_cls"),
    [
        ("catboost_classifier", "classifier", CatBoostClassifierWrapper),
        ("xgb_classifier", "classifier", XGBoostClassifierWrapper),
        ("catboost_tweedie_regressor", "regressor", CatBoostRegressorWrapper),
        ("xgb_regressor", "regressor", XGBoostRegressorWrapper),
        ("sklearn_tweedie_regressor", "regressor", SklearnRegressorWrapper),
        ("sklearn_gamma_regressor", "regressor", SklearnRegressorWrapper),
    ],
)
def test_get_model_wrapper_aliases(model_name, role, expected_cls):
    model = get_model_wrapper(
        model_name=model_name,
        role=role,
        random_seed=42,
        params={"iterations": 10} if "catboost" in model_name else {},
    )
    assert isinstance(model, expected_cls)


@pytest.mark.parametrize(
    ("cfg_file", "role", "expected_cls"),
    [
        ("catboost_classifier.yaml", "classifier", CatBoostClassifierWrapper),
        ("catboost_tweedie_regressor.yaml", "regressor", CatBoostRegressorWrapper),
        ("sklearn_tweedie_regressor.yaml", "regressor", SklearnRegressorWrapper),
        ("sklearn_gamma_regressor.yaml", "regressor", SklearnRegressorWrapper),
    ],
)
def test_all_model_configs_resolve_to_expected_wrapper_family(cfg_file, role, expected_cls):
    cfg = OmegaConf.load(MODEL_CFG_DIR / cfg_file)
    model = get_model_wrapper(
        model_name=cfg.name,
        role=role,
        random_seed=42,
        # Phase 6 scope is name-to-wrapper resolution (registry routing), not
        # validating every hyperparameter payload here.
        params={},
    )
    assert isinstance(model, expected_cls)


def test_unknown_model_name_error_lists_valid_keys():
    with pytest.raises(ValueError) as exc_info:
        get_model_wrapper(
            model_name="totally_unknown_model",
            role="regressor",
            random_seed=42,
            params={},
        )

    message = str(exc_info.value)
    assert "Valid model keys" in message
    for key in list_model_keys("regressor"):
        assert key in message


def test_unknown_role_error_lists_valid_roles():
    with pytest.raises(ValueError) as exc_info:
        resolve_model(role="role9", model_name="xgb")

    message = str(exc_info.value)
    assert "Valid roles" in message
    assert "classifier" in message
    assert "regressor" in message


def test_resolve_model_is_case_insensitive_for_role_and_name():
    registration = resolve_model(role="CLASSIFIER", model_name=" XGB_CLASSIFIER ")
    assert registration.family == "xgb"


def test_duplicate_registration_raises_with_normalized_key():
    with pytest.raises(ValueError) as exc_info:
        register_model(" classifier ", "CATBOOST_CLASSIFIER", "catboost", CatBoostClassifierWrapper)

    message = str(exc_info.value)
    assert "Duplicate model registration" in message
    assert "role='classifier'" in message
    assert "key='catboost_classifier'" in message


def test_all_model_config_names_are_registered():
    config_names = {
        str(OmegaConf.load(cfg_path).name).lower() for cfg_path in MODEL_CFG_DIR.glob("*.yaml")
    }
    registered_names = set(list_model_keys("classifier")) | set(list_model_keys("regressor"))
    missing = config_names - registered_names
    assert not missing, f"Model config names missing from registry: {sorted(missing)}"


def test_quiet_params_for_registry_models():
    cat_quiet = apply_quiet_params("catboost_classifier", {"iterations": 100})
    assert cat_quiet["logging_level"] == "Silent"

    xgb_quiet = apply_quiet_params("xgb_classifier", {"max_depth": 4})
    assert xgb_quiet["verbosity"] == 0

    sklearn_quiet = apply_quiet_params("sklearn_tweedie_regressor", {"alpha": 1.0})
    assert sklearn_quiet == {"alpha": 1.0}


def test_quiet_params_is_case_insensitive():
    cat_quiet = apply_quiet_params(" CATBOOST_CLASSIFIER ", {"iterations": 100})
    assert cat_quiet["logging_level"] == "Silent"
