import pytest
from fof8_ml.models.catboost_wrapper import CatBoostClassifierWrapper, CatBoostRegressorWrapper


class _DummyCBModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_catboost_constructors(monkeypatch):
    captured = {}

    def _classifier_ctor(**kwargs):
        captured["classifier"] = kwargs
        return _DummyCBModel(**kwargs)

    def _regressor_ctor(**kwargs):
        captured["regressor"] = kwargs
        return _DummyCBModel(**kwargs)

    monkeypatch.setattr("fof8_ml.models.catboost_wrapper.cb.CatBoostClassifier", _classifier_ctor)
    monkeypatch.setattr("fof8_ml.models.catboost_wrapper.cb.CatBoostRegressor", _regressor_ctor)
    return captured


def test_regressor_defaults_to_rmse_when_loss_function_not_provided(monkeypatch):
    captured = _patch_catboost_constructors(monkeypatch)

    CatBoostRegressorWrapper(random_seed=7, use_gpu=False)

    assert captured["regressor"]["loss_function"] == "RMSE"


@pytest.mark.parametrize("loss_function", ["Tweedie:variance_power=1.5", "Poisson"])
def test_regressor_preserves_configured_loss_function(monkeypatch, loss_function):
    captured = _patch_catboost_constructors(monkeypatch)

    CatBoostRegressorWrapper(random_seed=7, use_gpu=False, loss_function=loss_function)

    assert captured["regressor"]["loss_function"] == loss_function


def test_regressor_composes_tweedie_variance_power(monkeypatch):
    captured = _patch_catboost_constructors(monkeypatch)

    CatBoostRegressorWrapper(
        random_seed=7,
        use_gpu=False,
        loss_function="Tweedie",
        variance_power=1.7,
    )

    assert captured["regressor"]["loss_function"] == "Tweedie:variance_power=1.7"
    assert "variance_power" not in captured["regressor"]


def test_regressor_composes_expectile_alpha(monkeypatch):
    captured = _patch_catboost_constructors(monkeypatch)

    CatBoostRegressorWrapper(
        random_seed=7,
        use_gpu=False,
        loss_function="Expectile",
        expectile_alpha=0.7,
    )

    assert captured["regressor"]["loss_function"] == "Expectile:alpha=0.7"
    assert "expectile_alpha" not in captured["regressor"]


def test_regressor_preserves_configured_iterations(monkeypatch):
    captured = _patch_catboost_constructors(monkeypatch)

    CatBoostRegressorWrapper(random_seed=7, use_gpu=False, iterations=1234)

    assert captured["regressor"]["iterations"] == 1234


def test_gpu_does_not_force_devices_when_not_configured(monkeypatch):
    captured = _patch_catboost_constructors(monkeypatch)
    monkeypatch.setattr("fof8_ml.models.catboost_wrapper.torch.cuda.is_available", lambda: True)

    CatBoostClassifierWrapper(random_seed=7, use_gpu=True)
    CatBoostRegressorWrapper(random_seed=7, use_gpu=True)

    assert captured["classifier"]["task_type"] == "GPU"
    assert "devices" not in captured["classifier"]
    assert captured["regressor"]["task_type"] == "GPU"
    assert "devices" not in captured["regressor"]


@pytest.mark.parametrize("devices", ["0", 1])
def test_gpu_preserves_configured_devices(monkeypatch, devices):
    captured = _patch_catboost_constructors(monkeypatch)
    monkeypatch.setattr("fof8_ml.models.catboost_wrapper.torch.cuda.is_available", lambda: True)

    CatBoostClassifierWrapper(random_seed=7, use_gpu=True, devices=devices)
    CatBoostRegressorWrapper(random_seed=7, use_gpu=True, devices=devices)

    assert captured["classifier"]["devices"] == str(devices)
    assert captured["regressor"]["devices"] == str(devices)
