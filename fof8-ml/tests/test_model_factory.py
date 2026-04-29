import os
import sys

from fof8_ml.models import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    XGBoostClassifierWrapper,
)

# Ensure pipelines module is available
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))

from pipelines.train import get_model_wrapper  # noqa: E402


def test_get_model_wrapper_stage1_catboost():
    model = get_model_wrapper(
        model_name="s1_catboost", stage="stage1", random_seed=42, params={"iterations": 10}
    )
    assert isinstance(model, CatBoostClassifierWrapper)


def test_get_model_wrapper_stage1_xgboost():
    model = get_model_wrapper(
        model_name="xgb", stage="stage1", random_seed=42, params={"n_estimators": 10}
    )
    assert isinstance(model, XGBoostClassifierWrapper)


def test_get_model_wrapper_stage2_catboost():
    model = get_model_wrapper(
        model_name="s2_catboost", stage="stage2", random_seed=42, params={"iterations": 10}
    )
    assert isinstance(model, CatBoostRegressorWrapper)
