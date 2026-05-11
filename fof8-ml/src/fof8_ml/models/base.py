from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

import catboost as cb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import GammaRegressor, TweedieRegressor
from sklearn.neural_network import MLPRegressor

ModelRole: TypeAlias = Literal["classifier", "regressor"]
ModelFamily: TypeAlias = Literal["catboost", "xgb", "sklearn"]
SupportedClassifierModel: TypeAlias = cb.CatBoostClassifier | xgb.XGBClassifier
SupportedSklearnRegressorModel: TypeAlias = TweedieRegressor | GammaRegressor | MLPRegressor
SupportedRegressorModel: TypeAlias = (
    cb.CatBoostRegressor | xgb.XGBRegressor | SupportedSklearnRegressorModel
)
SupportedMLModel: TypeAlias = SupportedClassifierModel | SupportedRegressorModel
TModel = TypeVar("TModel", bound=SupportedMLModel)


class ModelWrapper(Generic[TModel], ABC):
    """Base class for all machine learning models in the pipeline."""

    def __init__(self, use_gpu: bool = False, **params: object) -> None:
        self.params: dict[str, Any] = cast(dict[str, Any], params)
        self.use_gpu = use_gpu
        self.model: TModel | None = None

    def require_model(self) -> TModel:
        """Return the initialized model, raising if accessed before construction."""
        assert self.model is not None
        return self.model

    @abstractmethod
    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
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
    def log_model(self, name: str, X: pl.DataFrame | None = None) -> None:
        """Log the model to MLflow."""
        pass

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the features before prediction or evaluation.
        Default implementation returns X unchanged.
        """
        return X

    @abstractmethod
    def get_feature_importance(self) -> tuple[list[str], np.ndarray]:
        """
        Return the feature names and their corresponding importance values.

        Returns:
            A tuple of (feature_names, importance_values).
        """
        pass
