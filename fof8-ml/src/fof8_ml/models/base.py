from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class ModelWrapper(ABC):
    """Base class for all machine learning models in the pipeline."""

    def __init__(self, use_gpu: bool = False, **params):
        self.params = params
        self.use_gpu = use_gpu
        self.model = None

    @abstractmethod
    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame = None,
        y_val: np.ndarray = None,
    ):
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Predict target values for X."""
        pass

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Predict class probabilities for X. Only implemented for classifiers."""
        raise NotImplementedError("predict_proba is not implemented for this model.")

    @abstractmethod
    def get_best_iteration(self) -> int:
        """Return the best iteration found during early stopping."""
        pass

    @abstractmethod
    def log_model(self, name: str):
        """Log the model to MLflow."""
        pass
