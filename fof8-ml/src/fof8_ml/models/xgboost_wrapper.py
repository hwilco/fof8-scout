from typing import Any, TypeVar

import mlflow
import numpy as np
import polars as pl
import torch
import xgboost as xgb
from mlflow.models import infer_signature

from .base import ModelWrapper

TXGBoostModel = TypeVar("TXGBoostModel", xgb.XGBClassifier, xgb.XGBRegressor)


class XGBoostWrapper(ModelWrapper[TXGBoostModel]):
    """Shared logic for XGBoost wrapper."""

    def get_best_iteration(self) -> int:
        return int(getattr(self.require_model(), "best_iteration", 0))

    def _signature_kwargs(self, X: pl.DataFrame | None) -> dict[str, Any]:
        if X is None:
            return {}
        input_example = X.head(5).to_pandas()
        try:
            prediction = self.require_model().predict(input_example)
            return {
                "input_example": input_example,
                "signature": infer_signature(input_example, prediction),
            }
        except Exception:
            return {"input_example": input_example}

    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        mlflow.xgboost.log_model(
            self.require_model(), artifact_path=name, **self._signature_kwargs(X)
        )

    def get_feature_importance(self) -> tuple[list[str], np.ndarray]:
        """Returns feature names and importance values from XGBoost."""
        model = self.require_model()
        return model.feature_names_in_.tolist(), np.asarray(model.feature_importances_)


class XGBoostClassifierWrapper(ModelWrapper[xgb.XGBClassifier]):
    def __init__(self, random_seed: int, use_gpu: bool = False, **params: object) -> None:
        super().__init__(use_gpu=use_gpu, **params)

        # GPU configuration
        if self.use_gpu and torch.cuda.is_available():
            self.params["device"] = "cuda"

        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 10)
        self.model = xgb.XGBClassifier(
            **self.params,
            enable_categorical=True,
            random_state=random_seed,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> None:
        _ = sample_weight, sample_weight_val
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.require_model().fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.require_model().predict(X))

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.require_model().predict_proba(X)[:, 1])

    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        if X is None:
            mlflow.xgboost.log_model(self.require_model(), artifact_path=name)
            return

        input_example = X.head(5).to_pandas()
        try:
            prediction = self.require_model().predict_proba(input_example)[:, 1]
            signature = infer_signature(input_example, prediction)
            mlflow.xgboost.log_model(
                self.require_model(),
                artifact_path=name,
                input_example=input_example,
                signature=signature,
            )
        except Exception:
            mlflow.xgboost.log_model(
                self.require_model(),
                artifact_path=name,
                input_example=input_example,
            )

    def get_best_iteration(self) -> int:
        return int(getattr(self.require_model(), "best_iteration", 0))

    def get_feature_importance(self) -> tuple[list[str], np.ndarray]:
        model = self.require_model()
        return model.feature_names_in_.tolist(), np.asarray(model.feature_importances_)


class XGBoostRegressorWrapper(ModelWrapper[xgb.XGBRegressor]):
    def __init__(self, random_seed: int, use_gpu: bool = False, **params: object) -> None:
        super().__init__(use_gpu=use_gpu, **params)
        clean_params = self.params.copy()

        # GPU configuration
        if self.use_gpu and torch.cuda.is_available():
            clean_params["device"] = "cuda"

        for key in ["objective", "eval_metric", "n_estimators"]:
            clean_params.pop(key, None)
        clean_params["objective"] = "reg:squarederror"

        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 10)
        self.model = xgb.XGBRegressor(
            **clean_params,
            enable_categorical=True,
            random_state=random_seed,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> None:
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        fit_kwargs: dict[str, Any] = {
            "eval_set": eval_set,
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None and sample_weight_val is not None:
            fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
        self.require_model().fit(X_train, y_train, **fit_kwargs)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.require_model().predict(X))

    def get_best_iteration(self) -> int:
        return int(getattr(self.require_model(), "best_iteration", 0))

    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        mlflow.xgboost.log_model(
            self.require_model(), artifact_path=name, **self._signature_kwargs(X)
        )

    def _signature_kwargs(self, X: pl.DataFrame | None) -> dict[str, Any]:
        if X is None:
            return {}
        input_example = X.head(5).to_pandas()
        try:
            prediction = self.require_model().predict(input_example)
            return {
                "input_example": input_example,
                "signature": infer_signature(input_example, prediction),
            }
        except Exception:
            return {"input_example": input_example}

    def get_feature_importance(self) -> tuple[list[str], np.ndarray]:
        model = self.require_model()
        return model.feature_names_in_.tolist(), np.asarray(model.feature_importances_)
