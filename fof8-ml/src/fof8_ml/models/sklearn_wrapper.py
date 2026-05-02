from typing import Any

import joblib
import mlflow
import numpy as np
import polars as pl
from sklearn.linear_model import GammaRegressor, TweedieRegressor

from fof8_ml.data.preprocessing import preprocess_for_sklearn

from .base import ModelWrapper


class SklearnRegressorWrapper(ModelWrapper):
    def __init__(self, model_name: str, **params: object) -> None:
        use_gpu = bool(params.pop("use_gpu", False))
        super().__init__(use_gpu=use_gpu, **params)
        self.scaler = None
        self.columns = None
        typed_params: dict[str, Any] = {k: v for k, v in self.params.items() if v is not None}

        if "tweedie" in model_name.lower():
            self.model = TweedieRegressor(**typed_params)
        else:
            self.model = GammaRegressor(**typed_params)

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        # GLMs expect strictly positive target for gamma/poisson/tweedie
        # In the pipeline, the target is already transformed by np.log1p.
        # But for sklearn, the pipeline actually passes np.expm1(y) in the inner loop!
        # Wait, the pipeline logic for sklearn does:
        # y_cv_train_raw = np.expm1(y_cv_train)
        # We will handle the target transform *inside* the wrapper for simplicity, OR
        # assume y_train passed to this wrapper is already the raw target.
        # In our refactor, we will pass the log-transformed target to ALL wrappers uniformly,
        # and let the wrapper invert it if needed.

        y_train_raw = np.expm1(y_train)

        X_sk, self.scaler, self.columns = preprocess_for_sklearn(X_train)
        self.model.fit(X_sk, y_train_raw)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_sk, _, _ = preprocess_for_sklearn(X, scaler=self.scaler, expected_columns=self.columns)

        y_pred_raw = self.model.predict(X_sk)

        # Convert back to log space to be consistent with tree models
        y_pred_log = np.log1p(np.maximum(y_pred_raw, 0))
        return y_pred_log

    def get_best_iteration(self) -> int:
        return 0

    def log_model(self, name: str) -> None:
        assert self.columns is not None
        mlflow.sklearn.log_model(self.model, name=name)
        joblib.dump(self.scaler, f"{name}_scaler.joblib")
        mlflow.log_artifact(f"{name}_scaler.joblib")

        with open(f"{name}_features.txt", "w") as f:
            f.write("\n".join(self.columns))
        mlflow.log_artifact(f"{name}_features.txt")

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply the same preprocessing used during training."""
        X_sk, _, _ = preprocess_for_sklearn(X, scaler=self.scaler, expected_columns=self.columns)
        return X_sk

    def get_feature_importance(self) -> tuple[list[str], np.ndarray]:
        """Returns the one-hot encoded feature names and absolute coefficients."""
        assert self.columns is not None
        if hasattr(self.model, "coef_"):
            return self.columns, np.abs(self.model.coef_)
        else:
            return self.columns, np.zeros(len(self.columns))
