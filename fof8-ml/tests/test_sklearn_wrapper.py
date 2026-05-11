import numpy as np
import pandas as pd
import polars as pl
from fof8_ml.models.sklearn_wrapper import SklearnMLPRegressorWrapper, SklearnRegressorWrapper


def test_sklearn_signature_kwargs_use_pandas_input_example():
    wrapper = SklearnRegressorWrapper(
        model_name="sklearn_tweedie_regressor", alpha=1.0, max_iter=100
    )
    X = pl.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["A", "B", "A"]})
    y_log = np.log1p(np.array([10.0, 20.0, 15.0]))
    wrapper.fit(X, y_log)

    signature_kwargs = wrapper._signature_kwargs(X)

    assert isinstance(signature_kwargs["input_example"], pd.DataFrame)
    assert "signature" in signature_kwargs


def test_sklearn_mlp_preprocesses_mixed_features_with_missing_indicators():
    wrapper = SklearnMLPRegressorWrapper(
        hidden_layer_sizes=(4,),
        max_iter=50,
        random_state=42,
        early_stopping=False,
    )
    X = pl.DataFrame(
        {
            "Mean_Accuracy": [80.0, None, 70.0, None, 65.0, 72.0],
            "Dash": [4.6, 4.8, 4.7, 4.9, 4.55, 4.75],
            "Position_Group": ["QB", "DE", "QB", "DE", "QB", "DE"],
        }
    )
    y = np.array([70.0, 55.0, 68.0, 58.0, 72.0, 57.0])

    wrapper.fit(X, y)
    transformed = wrapper.transform(X)

    assert "Mean_Accuracy__missing" in transformed.columns
    assert "Position_Group_QB" in transformed.columns
    assert "Position_Group_DE" in transformed.columns
    assert wrapper.numeric_medians["Mean_Accuracy"] == 71.0
    assert transformed.null_count().sum_horizontal().item() == 0


def test_sklearn_mlp_accepts_hidden_layer_size_aliases():
    wrapper = SklearnMLPRegressorWrapper(
        hidden_layer_sizes="8x4",
        max_iter=50,
        random_state=42,
        early_stopping=False,
    )

    assert wrapper.require_model().hidden_layer_sizes == (8, 4)


def test_sklearn_mlp_aligns_unseen_categories_and_predicts_raw_scale():
    wrapper = SklearnMLPRegressorWrapper(
        hidden_layer_sizes=(4,),
        max_iter=50,
        random_state=42,
        early_stopping=False,
    )
    X_train = pl.DataFrame(
        {
            "Mean_Accuracy": [80.0, None, 70.0, 65.0, 72.0, None],
            "Position_Group": ["QB", "DE", "QB", "QB", "DE", "DE"],
        }
    )
    y_raw = np.array([70.0, 55.0, 68.0, 72.0, 57.0, 58.0])
    wrapper.fit(X_train, y_raw)

    X_infer = pl.DataFrame(
        {
            "Mean_Accuracy": [75.0, None],
            "Position_Group": ["QB", "WR"],
        }
    )
    transformed = wrapper.transform(X_infer)
    preds = wrapper.predict(X_infer)

    assert transformed.columns == wrapper.columns
    assert "Position_Group_WR" not in transformed.columns
    assert preds.shape == (2,)
    assert not np.allclose(preds, np.log1p(preds))
