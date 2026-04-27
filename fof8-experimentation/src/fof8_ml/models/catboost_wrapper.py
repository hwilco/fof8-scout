import catboost as cb
import polars as pl
import pandas as pd
import numpy as np
import mlflow
import torch

from .base import ModelWrapper

class CatBoostWrapper(ModelWrapper):
    """Shared logic for CatBoost wrapper."""
    def _prepare_data(self, X: pl.DataFrame):
        X_pd = X.to_pandas()
        cat_features = X_pd.select_dtypes(include=['object', 'category']).columns.tolist()
        return X_pd, cat_features

    def get_best_iteration(self) -> int:
        return self.model.get_best_iteration()

    def log_model(self, name: str):
        mlflow.catboost.log_model(self.model, name=name)

class CatBoostClassifierWrapper(CatBoostWrapper):
    def __init__(self, random_seed: int, use_gpu: bool = False, thread_count: int = -1, **params):
        super().__init__(use_gpu=use_gpu, **params)
        self.thread_count = thread_count
        
        # GPU configuration
        if self.use_gpu and torch.cuda.is_available():
            self.params['task_type'] = 'GPU'
            self.params['devices'] = '0'
                        
        self.model = cb.CatBoostClassifier(
            **self.params,
            random_seed=random_seed,
            thread_count=self.thread_count,
            verbose=False
        )
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 10)

    def fit(self, X_train: pl.DataFrame, y_train: np.ndarray, X_val: pl.DataFrame = None, y_val: np.ndarray = None):
        X_train_pd, cat_features = self._prepare_data(X_train)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_pd, _ = self._prepare_data(X_val)
            eval_set = [(X_val_pd, y_val)]
            
        self.model.fit(
            X_train_pd, y_train,
            eval_set=eval_set,
            cat_features=cat_features,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None
        )

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_pd, _ = self._prepare_data(X)
        return self.model.predict(X_pd)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        X_pd, _ = self._prepare_data(X)
        return self.model.predict_proba(X_pd)[:, 1]

class CatBoostRegressorWrapper(CatBoostWrapper):
    def __init__(self, random_seed: int, use_gpu: bool = False, thread_count: int = -1, **params):
        super().__init__(use_gpu=use_gpu, **params)
        self.thread_count = thread_count
        # Clean params for regressor
        clean_params = self.params.copy()
        
        # GPU configuration
        if self.use_gpu and torch.cuda.is_available():
            clean_params['task_type'] = 'GPU'
            clean_params['devices'] = '0'

        for key in ['loss_function', 'eval_metric', 'iterations', 'auto_class_weights']:
            clean_params.pop(key, None)
        clean_params['loss_function'] = 'RMSE'

        self.model = cb.CatBoostRegressor(
            **clean_params,
            random_seed=random_seed,
            thread_count=self.thread_count,
            verbose=False
        )
        self.early_stopping_rounds = self.params.get("early_stopping_rounds", 10)

    def fit(self, X_train: pl.DataFrame, y_train: np.ndarray, X_val: pl.DataFrame = None, y_val: np.ndarray = None):
        X_train_pd, cat_features = self._prepare_data(X_train)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_pd, _ = self._prepare_data(X_val)
            eval_set = [(X_val_pd, y_val)]
            
        self.model.fit(
            X_train_pd, y_train,
            eval_set=eval_set,
            cat_features=cat_features,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None
        )

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_pd, _ = self._prepare_data(X)
        return self.model.predict(X_pd)
