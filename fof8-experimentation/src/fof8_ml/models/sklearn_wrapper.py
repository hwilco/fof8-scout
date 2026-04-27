from sklearn.linear_model import TweedieRegressor, GammaRegressor
import polars as pl
import numpy as np
import mlflow
import joblib
import os

from .base import ModelWrapper
from fof8_ml.data.transforms import preprocess_for_sklearn

class SklearnRegressorWrapper(ModelWrapper):
    def __init__(self, model_name: str, **params):
        super().__init__(**params)
        self.scaler = None
        self.columns = None
        
        if "tweedie" in model_name.lower():
            self.model = TweedieRegressor(**self.params)
        else:
            self.model = GammaRegressor(**self.params)

    def fit(self, X_train: pl.DataFrame, y_train: np.ndarray, X_val: pl.DataFrame = None, y_val: np.ndarray = None):
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
        
        X_sk, self.scaler = preprocess_for_sklearn(X_train)
        self.columns = X_sk.columns
        self.model.fit(X_sk, y_train_raw)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_sk, _ = preprocess_for_sklearn(X, scaler=self.scaler)
        X_sk = X_sk.reindex(columns=self.columns, fill_value=0)
        y_pred_raw = self.model.predict(X_sk)
        
        # Convert back to log space to be consistent with tree models
        y_pred_log = np.log1p(np.maximum(y_pred_raw, 0))
        return y_pred_log

    def get_best_iteration(self) -> int:
        return 0

    def log_model(self, name: str):
        mlflow.sklearn.log_model(self.model, name=name)
        joblib.dump(self.scaler, f"{name}_scaler.joblib")
        mlflow.log_artifact(f"{name}_scaler.joblib")
        
        with open(f"{name}_features.txt", "w") as f:
            f.write("\n".join(self.columns.tolist()))
        mlflow.log_artifact(f"{name}_features.txt")
