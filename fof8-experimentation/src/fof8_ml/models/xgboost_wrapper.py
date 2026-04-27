import xgboost as xgb
import polars as pl
import pandas as pd
import numpy as np
import mlflow
import torch

from .base import ModelWrapper


class XGBoostWrapper(ModelWrapper):
    """Shared logic for XGBoost wrapper."""

    def get_best_iteration(self) -> int:
        return getattr(self.model, "best_iteration", 0)

    def log_model(self, name: str):
        mlflow.xgboost.log_model(self.model, name=name)


class XGBoostClassifierWrapper(XGBoostWrapper):
    def __init__(self, random_seed: int, use_gpu: bool = False, **params):
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
        X_val: pl.DataFrame = None,
        y_val: np.ndarray = None,
    ):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class XGBoostRegressorWrapper(XGBoostWrapper):
    def __init__(self, random_seed: int, use_gpu: bool = False, **params):
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
        X_val: pl.DataFrame = None,
        y_val: np.ndarray = None,
    ):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self.model.predict(X)
