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
    ("model_name", "stage", "expected_cls"),
    [
        ("s1_catboost", "stage1", CatBoostClassifierWrapper),
        ("xgb", "stage1", XGBoostClassifierWrapper),
        ("s2_catboost", "stage2", CatBoostRegressorWrapper),
        ("s2_xgb", "stage2", XGBoostRegressorWrapper),
        ("reg_tweedie", "stage2", SklearnRegressorWrapper),
        ("reg_gamma", "stage2", SklearnRegressorWrapper),
    ],
)
def test_get_model_wrapper_aliases(model_name, stage, expected_cls):
    model = get_model_wrapper(
        model_name=model_name,
        stage=stage,
        random_seed=42,
        params={"iterations": 10} if "catboost" in model_name else {},
    )
    assert isinstance(model, expected_cls)


@pytest.mark.parametrize(
    ("cfg_file", "stage", "expected_cls"),
    [
        ("s1_catboost.yaml", "stage1", CatBoostClassifierWrapper),
        ("career_threshold_catboost.yaml", "stage1", CatBoostClassifierWrapper),
        ("career_threshold_catboost.yaml", "stage2", CatBoostRegressorWrapper),
        ("career_threshold_xgb.yaml", "stage1", XGBoostClassifierWrapper),
        ("career_threshold_xgb.yaml", "stage2", XGBoostRegressorWrapper),
        ("reg_tweedie.yaml", "stage2", SklearnRegressorWrapper),
        ("reg_gamma.yaml", "stage2", SklearnRegressorWrapper),
    ],
)
def test_all_model_configs_resolve_to_expected_wrapper_family(cfg_file, stage, expected_cls):
    cfg = OmegaConf.load(MODEL_CFG_DIR / cfg_file)
    model = get_model_wrapper(
        model_name=cfg.name,
        stage=stage,
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
            stage="stage2",
            random_seed=42,
            params={},
        )

    message = str(exc_info.value)
    assert "Valid model keys" in message
    for key in list_model_keys("stage2"):
        assert key in message


def test_unknown_stage_error_lists_valid_stages():
    with pytest.raises(ValueError) as exc_info:
        resolve_model(stage="stage9", model_name="xgb")

    message = str(exc_info.value)
    assert "Valid stages" in message
    assert "stage1" in message
    assert "stage2" in message


def test_resolve_model_is_case_insensitive_for_stage_and_name():
    registration = resolve_model(stage="STAGE1", model_name=" XGB ")
    assert registration.family == "xgb"


def test_duplicate_registration_raises_with_normalized_key():
    with pytest.raises(ValueError) as exc_info:
        register_model(" stage1 ", "S1_CATBOOST", "catboost", CatBoostClassifierWrapper)

    message = str(exc_info.value)
    assert "Duplicate model registration" in message
    assert "stage='stage1'" in message
    assert "key='s1_catboost'" in message


def test_all_model_config_names_are_registered():
    config_names = {
        str(OmegaConf.load(cfg_path).name).lower() for cfg_path in MODEL_CFG_DIR.glob("*.yaml")
    }
    registered_names = set(list_model_keys("stage1")) | set(list_model_keys("stage2"))
    missing = config_names - registered_names
    assert not missing, f"Model config names missing from registry: {sorted(missing)}"


def test_quiet_params_for_registry_models():
    cat_quiet = apply_quiet_params("catboost_career_threshold", {"iterations": 100})
    assert cat_quiet["logging_level"] == "Silent"

    xgb_quiet = apply_quiet_params("xgboost_career_threshold", {"max_depth": 4})
    assert xgb_quiet["verbosity"] == 0

    sklearn_quiet = apply_quiet_params("sklearn_tweedie", {"alpha": 1.0})
    assert sklearn_quiet == {"alpha": 1.0}


def test_quiet_params_is_case_insensitive():
    cat_quiet = apply_quiet_params(" CATBOOST_CAREER_THRESHOLD ", {"iterations": 100})
    assert cat_quiet["logging_level"] == "Silent"
