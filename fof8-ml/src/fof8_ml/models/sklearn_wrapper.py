from typing import Any

import joblib
import mlflow
import numpy as np
import polars as pl
from mlflow.models import infer_signature
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

        y_train_raw = np.expm1(y_train)

        X_sk, self.scaler, self.columns = preprocess_for_sklearn(X_train)
        self.model.fit(X_sk.to_numpy(), y_train_raw)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_sk, _, _ = preprocess_for_sklearn(X, scaler=self.scaler, expected_columns=self.columns)

        y_pred_raw = self.model.predict(X_sk.to_numpy())

        # Convert back to log space to be consistent with tree models
        y_pred_log = np.log1p(np.maximum(y_pred_raw, 0))
        return y_pred_log

    def get_best_iteration(self) -> int:
        return 0

    def _signature_kwargs(self, X: pl.DataFrame | None) -> dict[str, Any]:
        signature_kwargs: dict[str, Any] = {}
        if X is not None:
            input_example = self.transform(X.head(5)).to_pandas()
            try:
                prediction = self.model.predict(input_example.to_numpy())
                signature_kwargs = {
                    "input_example": input_example,
                    "signature": infer_signature(input_example, prediction),
                }
            except Exception:
                signature_kwargs = {"input_example": input_example}
        return signature_kwargs

    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        assert self.columns is not None
        mlflow.sklearn.log_model(self.model, artifact_path=name, **self._signature_kwargs(X))
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
