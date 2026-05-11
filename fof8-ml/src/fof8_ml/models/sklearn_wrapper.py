from numbers import Real
from typing import Any

import joblib
import mlflow
import numpy as np
import polars as pl
from mlflow.models import infer_signature
from sklearn.linear_model import GammaRegressor, TweedieRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from fof8_ml.data.preprocessing import preprocess_for_sklearn
from fof8_ml.timing import timed_step

from .base import ModelWrapper, SupportedSklearnRegressorModel


class SklearnRegressorWrapper(ModelWrapper[SupportedSklearnRegressorModel]):
    def __init__(self, model_name: str, **params: object) -> None:
        use_gpu = bool(params.pop("use_gpu", False))
        params.pop("random_seed", None)
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
        self.require_model().fit(X_sk.to_numpy(), y_train_raw)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_sk, _, _ = preprocess_for_sklearn(X, scaler=self.scaler, expected_columns=self.columns)

        y_pred_raw = self.require_model().predict(X_sk.to_numpy())

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
                prediction = self.require_model().predict(input_example.to_numpy())
                signature_kwargs = {
                    "input_example": input_example,
                    "signature": infer_signature(input_example, prediction),
                }
            except Exception:
                signature_kwargs = {"input_example": input_example}
        return signature_kwargs

    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        assert self.columns is not None
        mlflow.sklearn.log_model(
            self.require_model(), artifact_path=name, **self._signature_kwargs(X)
        )
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
        model = self.require_model()
        coefficients = getattr(model, "coef_", None)
        return self.columns, np.abs(coefficients) if coefficients is not None else np.zeros(
            len(self.columns)
        )


class SklearnMLPRegressorWrapper(ModelWrapper[MLPRegressor]):
    """sklearn MLP regressor with MLP-specific dense preprocessing.

    This intentionally does not reuse the GLM sklearn preprocessing because masked
    position-skill nulls need explicit missingness indicators before imputation.
    """

    def __init__(self, model_name: str = "sklearn_mlp_regressor", **params: object) -> None:
        use_gpu = bool(params.pop("use_gpu", False))
        self.timing_diagnostics = bool(params.pop("timing_diagnostics", False))
        random_seed = params.pop("random_seed", None)
        super().__init__(use_gpu=use_gpu, **params)
        self.scaler: StandardScaler | None = None
        self.columns: list[str] | None = None
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.missing_indicator_columns: list[str] = []
        self.numeric_medians: dict[str, float] = {}

        typed_params: dict[str, Any] = {k: v for k, v in self.params.items() if v is not None}
        if "hidden_layer_sizes" in typed_params:
            typed_params["hidden_layer_sizes"] = self._coerce_hidden_layer_sizes(
                typed_params["hidden_layer_sizes"]
            )
        if random_seed is not None and "random_state" not in typed_params:
            typed_params["random_state"] = int(str(random_seed))
        self.model = MLPRegressor(**typed_params)

    @staticmethod
    def _coerce_hidden_layer_sizes(value: object) -> object:
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
            return tuple(int(part) for part in parts)
        if isinstance(value, list):
            return tuple(int(part) for part in value)
        return value

    @staticmethod
    def _categorical_columns(X: pl.DataFrame) -> list[str]:
        return [
            col for col, dtype in X.schema.items() if dtype in [pl.String, pl.Categorical, pl.Enum]
        ]

    def _fit_preprocess(self, X: pl.DataFrame) -> np.ndarray:
        self.categorical_columns = self._categorical_columns(X)
        self.numeric_columns = [col for col in X.columns if col not in self.categorical_columns]

        numeric_exprs = []
        indicator_exprs = []
        self.missing_indicator_columns = []
        self.numeric_medians = {}

        for col in self.numeric_columns:
            median = X.get_column(col).median()
            fill_value = float(median) if isinstance(median, Real) else 0.0
            self.numeric_medians[col] = fill_value
            numeric_exprs.append(pl.col(col).cast(pl.Float64).fill_null(fill_value).alias(col))
            if X.get_column(col).null_count() > 0:
                indicator_col = f"{col}__missing"
                self.missing_indicator_columns.append(indicator_col)
                indicator_exprs.append(pl.col(col).is_null().cast(pl.Int8).alias(indicator_col))

        frames: list[pl.DataFrame] = []
        if numeric_exprs:
            frames.append(X.select(numeric_exprs))
        if indicator_exprs:
            frames.append(X.select(indicator_exprs))
        if self.categorical_columns:
            frames.append(X.select(self.categorical_columns).to_dummies(drop_first=False))

        X_prepared = pl.concat(frames, how="horizontal") if frames else pl.DataFrame()
        self.columns = X_prepared.columns
        self.scaler = StandardScaler()
        return np.asarray(self.scaler.fit_transform(X_prepared.to_numpy()))

    def _transform_preprocess(self, X: pl.DataFrame) -> np.ndarray:
        if self.columns is None or self.scaler is None:
            raise RuntimeError("SklearnMLPRegressorWrapper must be fitted before transform.")

        numeric_exprs = [
            pl.col(col).cast(pl.Float64).fill_null(self.numeric_medians.get(col, 0.0)).alias(col)
            for col in self.numeric_columns
            if col in X.columns
        ]
        indicator_exprs = [
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}__missing")
            for col in self.numeric_columns
            if f"{col}__missing" in self.missing_indicator_columns and col in X.columns
        ]

        frames: list[pl.DataFrame] = []
        if numeric_exprs:
            frames.append(X.select(numeric_exprs))
        if indicator_exprs:
            frames.append(X.select(indicator_exprs))
        existing_categoricals = [col for col in self.categorical_columns if col in X.columns]
        if existing_categoricals:
            frames.append(X.select(existing_categoricals).to_dummies(drop_first=False))

        X_prepared = pl.concat(frames, how="horizontal") if frames else pl.DataFrame()
        missing_cols = [col for col in self.columns if col not in X_prepared.columns]
        if missing_cols:
            X_prepared = X_prepared.with_columns([pl.lit(0).alias(col) for col in missing_cols])
        X_prepared = X_prepared.select(self.columns)
        return np.asarray(self.scaler.transform(X_prepared.to_numpy()))

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        _ = X_val, y_val
        with timed_step("sklearn_mlp.fit_preprocess", enabled=self.timing_diagnostics):
            X_np = self._fit_preprocess(X_train)
        with timed_step("sklearn_mlp.fit_model", enabled=self.timing_diagnostics):
            self.require_model().fit(X_np, y_train)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        with timed_step("sklearn_mlp.predict_preprocess", enabled=self.timing_diagnostics):
            X_np = self._transform_preprocess(X)
        with timed_step("sklearn_mlp.predict_model", enabled=self.timing_diagnostics):
            return np.asarray(self.require_model().predict(X_np))

    def get_best_iteration(self) -> int:
        model = self.require_model()
        return int(getattr(model, "n_iter_", 0))

    def _signature_kwargs(self, X: pl.DataFrame | None) -> dict[str, Any]:
        signature_kwargs: dict[str, Any] = {}
        if X is not None:
            with timed_step("sklearn_mlp.signature_transform", enabled=self.timing_diagnostics):
                input_example = self.transform(X.head(5)).to_pandas()
            try:
                with timed_step("sklearn_mlp.signature_predict", enabled=self.timing_diagnostics):
                    prediction = np.asarray(self.require_model().predict(input_example.to_numpy()))
                signature_kwargs = {
                    "input_example": input_example,
                    "signature": infer_signature(input_example, prediction),
                }
            except Exception:
                signature_kwargs = {"input_example": input_example}
        return signature_kwargs

    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        assert self.columns is not None
        with timed_step("sklearn_mlp.log_model.signature_kwargs", enabled=self.timing_diagnostics):
            signature_kwargs = self._signature_kwargs(X)
        with timed_step(
            "sklearn_mlp.log_model.mlflow_sklearn_log_model",
            enabled=self.timing_diagnostics,
        ):
            mlflow.sklearn.log_model(self.require_model(), artifact_path=name, **signature_kwargs)
        with timed_step("sklearn_mlp.log_model.preprocessor", enabled=self.timing_diagnostics):
            joblib.dump(
                {
                    "scaler": self.scaler,
                    "columns": self.columns,
                    "categorical_columns": self.categorical_columns,
                    "numeric_columns": self.numeric_columns,
                    "missing_indicator_columns": self.missing_indicator_columns,
                    "numeric_medians": self.numeric_medians,
                },
                f"{name}_preprocessor.joblib",
            )
            mlflow.log_artifact(f"{name}_preprocessor.joblib")

        with timed_step("sklearn_mlp.log_model.features", enabled=self.timing_diagnostics):
            with open(f"{name}_features.txt", "w") as f:
                f.write("\n".join(self.columns))
            mlflow.log_artifact(f"{name}_features.txt")

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        with timed_step("sklearn_mlp.transform_preprocess", enabled=self.timing_diagnostics):
            X_np = self._transform_preprocess(X)
        assert self.columns is not None
        return pl.DataFrame(X_np, schema=self.columns)

    def get_feature_importance(self) -> tuple[list[str], np.ndarray]:
        assert self.columns is not None
        return self.columns, np.zeros(len(self.columns))
