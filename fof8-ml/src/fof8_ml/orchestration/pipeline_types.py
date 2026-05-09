from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import polars as pl


@dataclass
class TimelineInfo:
    initial_year: int
    final_sim_year: int
    valid_start_year: int
    valid_end_year: int
    train_year_range: List[int]
    val_year_range: List[int]
    test_year_range: List[int]
    universes: List[str] | None = None
    per_universe: Dict[str, Dict[str, Any]] | None = None
    split_strategy: str | None = None
    split_unit: str | None = None


@dataclass
class PreparedData:
    """Output of DataLoader — everything downstream needs."""

    X_train: pl.DataFrame
    X_val: pl.DataFrame
    X_test: pl.DataFrame
    y_cls: np.ndarray  # classifier binary target
    y_cls_val: np.ndarray
    y_cls_test: np.ndarray
    y_reg: np.ndarray  # regressor continuous target
    y_reg_val: np.ndarray
    y_reg_test: np.ndarray
    meta_train: pl.DataFrame
    meta_val: pl.DataFrame
    meta_test: pl.DataFrame
    timeline: TimelineInfo  # year ranges, buffer, etc.
    metadata_columns: List[str]
    target_columns: List[str]
    outcomes_train: pl.DataFrame | None = None
    outcomes_val: pl.DataFrame | None = None
    outcomes_test: pl.DataFrame | None = None


@dataclass
class CVResult:
    """Output of a cross-validation run."""

    oof_predictions: np.ndarray
    best_iterations: List[int]
    fold_metrics: List[Dict[str, float]]


@dataclass
class ClassifierResult:
    """Complete classifier output, consumed by logging."""

    cv_result: CVResult
    calibrated_oof_probs: np.ndarray
    raw_oof_probs: np.ndarray
    optimal_threshold: float
    final_predictions: np.ndarray
    metrics: Dict[str, float]
    calibrator: Any  # BetaCalibrator type
