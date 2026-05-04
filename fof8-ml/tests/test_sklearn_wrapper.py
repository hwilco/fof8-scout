import numpy as np
import pandas as pd
import polars as pl

from fof8_ml.models.sklearn_wrapper import SklearnRegressorWrapper


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
