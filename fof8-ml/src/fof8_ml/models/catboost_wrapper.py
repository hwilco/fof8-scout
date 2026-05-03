from typing import Any

import catboost as cb
import mlflow
import numpy as np
import polars as pl
import torch

from .base import ModelWrapper


class CatBoostWrapper(ModelWrapper):
    """Shared logic for CatBoost wrapper."""

    def _prepare_data(self, X: pl.DataFrame) -> tuple[Any, list[str]]:
        """
        Converts Polars DataFrame to Pandas and identifies categorical features.

        Args:
            X: Input Polars DataFrame.

        Returns:
            Tuple of (Pandas DataFrame, list of categorical feature names).
        """
        X_pd = X.to_pandas()
        cat_features = X_pd.select_dtypes(include=["object", "category"]).columns.tolist()
        return X_pd, cat_features

    def get_best_iteration(self) -> int:
        assert self.model is not None
        return self.model.get_best_iteration()

    def log_model(self, name: str) -> None:
        assert self.model is not None
        mlflow.catboost.log_model(self.model, name=name)

    def get_feature_importance(self) -> tuple[list[str], np.ndarray]:
        """Returns feature names and importance values from CatBoost."""
        assert self.model is not None
        return self.model.feature_names_, self.model.get_feature_importance()


class CatBoostClassifierWrapper(CatBoostWrapper):
    def __init__(
        self,
        random_seed: int,
        use_gpu: bool = False,
        thread_count: int = -1,
        **params: object,
    ) -> None:
        super().__init__(use_gpu=use_gpu, **params)
        params_dict = self.params
        self.thread_count = thread_count

        # GPU configuration
        if self.use_gpu and torch.cuda.is_available():
            params_dict["task_type"] = "GPU"
            if "devices" in params_dict:
                params_dict["devices"] = str(params_dict["devices"])

        # Prevent verbosity conflict
        if not any(
            k in params_dict for k in ["verbose", "logging_level", "verbose_eval", "silent"]
        ):
            params_dict["verbose"] = False

        self.model = cb.CatBoostClassifier(
            **(params_dict), random_seed=random_seed, thread_count=self.thread_count
        )
        self.early_stopping_rounds = params_dict.pop("early_stopping_rounds", 10)

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        X_train_pd, cat_features = self._prepare_data(X_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_pd, _ = self._prepare_data(X_val)
            eval_set = [(X_val_pd, y_val)]

        self.model.fit(
            X_train_pd,
            y_train,
            eval_set=eval_set,
            cat_features=cat_features,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
        )

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_pd, _ = self._prepare_data(X)
        return self.model.predict(X_pd)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        X_pd, _ = self._prepare_data(X)
        return self.model.predict_proba(X_pd)[:, 1]


class CatBoostRegressorWrapper(CatBoostWrapper):
    def __init__(
        self,
        random_seed: int,
        use_gpu: bool = False,
        thread_count: int = -1,
        **params: object,
    ) -> None:
        super().__init__(use_gpu=use_gpu, **params)
        self.thread_count = thread_count
        # Clean params for regressor
        clean_params = self.params.copy()

        # GPU configuration
        if self.use_gpu and torch.cuda.is_available():
            clean_params["task_type"] = "GPU"
            if "devices" in clean_params:
                clean_params["devices"] = str(clean_params["devices"])

        for key in ["eval_metric", "iterations", "auto_class_weights"]:
            clean_params.pop(key, None)
        clean_params.setdefault("loss_function", "RMSE")

        # Prevent verbosity conflict
        if not any(
            k in clean_params for k in ["verbose", "logging_level", "verbose_eval", "silent"]
        ):
            clean_params["verbose"] = False

        self.model = cb.CatBoostRegressor(
            **(clean_params), random_seed=random_seed, thread_count=self.thread_count
        )
        self.early_stopping_rounds = self.params.get("early_stopping_rounds", 10)

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        X_train_pd, cat_features = self._prepare_data(X_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_pd, _ = self._prepare_data(X_val)
            eval_set = [(X_val_pd, y_val)]

        self.model.fit(
            X_train_pd,
            y_train,
            eval_set=eval_set,
            cat_features=cat_features,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
        )

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_pd, _ = self._prepare_data(X)
        return self.model.predict(X_pd)
