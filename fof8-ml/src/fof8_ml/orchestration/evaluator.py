import warnings
from collections.abc import Mapping

import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import (
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fof8_ml.evaluation.metrics import (
    calibration_slope,
    mean_ndcg_by_group,
    mean_topk_bias_by_group,
    mean_topk_weighted_mae_by_group,
    mean_topk_weighted_mae_normalized_by_group,
    topk_bias,
    topk_weighted_mae,
    topk_weighted_mae_normalized,
)

ConfigLike = DictConfig | Mapping[str, object] | None


def optimize_threshold(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    min_positive_recall: float = 0.0,
) -> tuple[float, float]:
    """Find threshold maximizing F1(bust) subject to min recall constraint.

    Args:
        y_true: Ground truth binary labels.
        calibrated_probs: Predicted probabilities.
        min_positive_recall: Minimum required recall for the positive class.

    Returns:
        (optimal_threshold, best_f1_bust)
    """
    best_threshold = 0.5
    best_f1_0 = -1.0

    thresholds = np.linspace(0.01, 0.99, 99)
    for thresh in thresholds:
        current_preds = (calibrated_probs >= thresh).astype(int)
        f1_0 = f1_score(y_true, current_preds, pos_label=0)
        recall_1 = recall_score(y_true, current_preds)

        if recall_1 >= min_positive_recall:
            if f1_0 > best_f1_0:
                best_f1_0 = f1_0
                best_threshold = thresh

    if best_f1_0 == -1.0:
        # No threshold satisfies the requested minimum positive recall.
        # Fall back deterministically to the threshold that yields maximum
        # positive recall, then best bust F1, then the lowest threshold.
        candidates: list[tuple[float, float, float]] = []
        for thresh in thresholds:
            current_preds = (calibrated_probs >= thresh).astype(int)
            recall_1 = recall_score(y_true, current_preds)
            f1_0 = f1_score(y_true, current_preds, pos_label=0)
            candidates.append((recall_1, f1_0, thresh))

        best_recall, best_f1_0, best_threshold = max(
            candidates, key=lambda item: (item[0], item[1], -item[2])
        )
        warnings.warn(
            (
                "No threshold satisfied min_positive_recall="
                f"{min_positive_recall:.4f}; using deterministic fallback threshold="
                f"{best_threshold:.2f} with achieved_recall={best_recall:.4f}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    return float(best_threshold), float(best_f1_0)


def compute_classifier_final_metrics(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute all classifier final metrics (bust precision, hit recall, PR-AUC, etc.)."""
    final_preds = (calibrated_probs >= threshold).astype(int)

    busts_filtered = np.sum((y_true == 0) & (final_preds == 0))
    hit_recall = recall_score(y_true, final_preds)
    bust_precision = precision_score(y_true, final_preds, pos_label=0)
    bust_recall = recall_score(y_true, final_preds, pos_label=0)

    p, r, _ = precision_recall_curve(y_true, calibrated_probs)
    pr_auc = auc(r, p)
    roc_auc = roc_auc_score(y_true, calibrated_probs)

    # Compute F1 for busts as well for convenience
    f1_bust = f1_score(y_true, final_preds, pos_label=0)

    return {
        "classifier_oof_busts_filtered": float(busts_filtered),
        "classifier_oof_hit_recall": float(hit_recall),
        "classifier_oof_f1_bust": float(f1_bust),
        "classifier_oof_precision_bust": float(bust_precision),
        "classifier_oof_recall_bust": float(bust_recall),
        "classifier_oof_pr_auc": float(pr_auc),
        "classifier_oof_roc_auc": float(roc_auc),
    }


def rename_metric_prefix(
    metrics: dict[str, float],
    source_prefix: str,
    target_prefix: str,
) -> dict[str, float]:
    """Return a copy of `metrics` with a metric-name prefix replaced."""
    renamed: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith(source_prefix):
            renamed[key.replace(source_prefix, target_prefix, 1)] = value
        else:
            renamed[key] = value
    return renamed


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Prefix metric names without altering the original metric body."""
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _cfg_get(config: ConfigLike, key: str, default: object = None) -> object:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _cfg_get_str(config: ConfigLike, key: str, default: str) -> str:
    value = _cfg_get(config, key, default)
    return default if value is None else str(value)


def _cfg_get_float(config: ConfigLike, key: str, default: float) -> float:
    value = _cfg_get(config, key, default)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _cfg_get_int(config: ConfigLike, key: str, default: int) -> int:
    value = _cfg_get(config, key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def fit_elite_thresholds(
    outcome_columns: pl.DataFrame | None,
    meta_columns: pl.DataFrame | None,
    elite_cfg: ConfigLike,
) -> dict[str, float] | None:
    """Fit elite thresholds from training outcomes for later holdout application.

    Args:
        outcome_columns: Training outcome columns containing the elite source outcome.
        meta_columns: Training metadata columns containing the scope column if grouped.
        elite_cfg: Config-like object describing elite threshold settings.

    Returns:
        Threshold mapping keyed by scope value, or `None` when elite derivation is disabled
        or cannot be fit from the available inputs.
    """
    if not bool(_cfg_get(elite_cfg, "enabled", False)) or outcome_columns is None:
        return None

    source_column = _cfg_get(elite_cfg, "source_column")
    if source_column is None or str(source_column) not in outcome_columns.columns:
        return None

    quantile = _cfg_get_float(elite_cfg, "quantile", 0.95)
    scope = _cfg_get_str(elite_cfg, "scope", "global")
    scope_column = _cfg_get_str(elite_cfg, "scope_column", "Position_Group")
    fallback_scope = _cfg_get_str(elite_cfg, "fallback_scope", "global")
    min_group_size = _cfg_get_int(elite_cfg, "min_group_size", 100)

    y_source = np.maximum(outcome_columns[str(source_column)].to_numpy().astype(float), 0.0)
    global_threshold = float(np.quantile(y_source, quantile)) if y_source.size else 0.0

    if scope == "global":
        return {"__global__": global_threshold}

    if meta_columns is None or scope_column not in meta_columns.columns:
        return {"__global__": global_threshold}

    thresholds: dict[str, float] = {"__global__": global_threshold}
    scope_values = meta_columns.get_column(scope_column).cast(pl.Utf8).to_numpy()
    for scope_value in np.unique(scope_values):
        mask = scope_values == scope_value
        if int(np.sum(mask)) < min_group_size:
            continue
        thresholds[str(scope_value)] = float(np.quantile(y_source[mask], quantile))

    if fallback_scope != "global":
        thresholds.setdefault("__global__", global_threshold)
    return thresholds


def compute_regressor_oof_metrics(
    y_true: np.ndarray,
    oof_predictions: np.ndarray,
    target_space: str = "log",
    draft_group: np.ndarray | None = None,
    draft_year: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute regressor OOF metrics in original target space.

    Args:
        y_true: Ground truth continuous labels in `target_space`.
        oof_predictions: Predicted continuous labels in `target_space`.
        target_space: Either "log" for log1p labels/predictions or "raw" for
            labels/predictions already in original target units.
        draft_group: Optional group key for draft-class-aware ranking/error metrics.
        draft_year: Legacy alias for `draft_group`.
    """
    if target_space == "log":
        y_real = np.expm1(y_true)
        y_pred = np.expm1(oof_predictions)
    elif target_space == "raw":
        y_real = y_true
        y_pred = np.maximum(oof_predictions, 0)
    else:
        raise ValueError(f"Unsupported target_space '{target_space}'. Expected 'log' or 'raw'.")

    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    mae = float(mean_absolute_error(y_real, y_pred))
    metrics = {
        "regressor_oof_rmse": rmse,
        "regressor_oof_mae": mae,
    }

    group_key = draft_group if draft_group is not None else draft_year
    if group_key is None:
        group_key = np.full(y_real.shape, "all", dtype=object)

    if draft_group is not None or draft_year is not None:
        top64_weighted_mae = mean_topk_weighted_mae_by_group(y_real, y_pred, group_key, k=64)
        top64_weighted_mae_normalized = mean_topk_weighted_mae_normalized_by_group(
            y_real, y_pred, group_key, k=64
        )
        top64_bias = mean_topk_bias_by_group(y_real, y_pred, group_key, k=64)
    else:
        top64_weighted_mae = topk_weighted_mae(y_real, y_pred, k=64)
        top64_weighted_mae_normalized = topk_weighted_mae_normalized(y_real, y_pred, k=64)
        top64_bias = topk_bias(y_real, y_pred, k=64)

    metrics.update(
        {
            "regressor_mean_ndcg_at_32": mean_ndcg_by_group(y_real, y_pred, group_key, k=32),
            "regressor_mean_ndcg_at_64": mean_ndcg_by_group(y_real, y_pred, group_key, k=64),
            "regressor_mean_ndcg_at_128": mean_ndcg_by_group(y_real, y_pred, group_key, k=128),
            "regressor_top64_weighted_mae": top64_weighted_mae,
            "regressor_top64_weighted_mae_normalized": top64_weighted_mae_normalized,
            "regressor_top64_bias": top64_bias,
            "regressor_top64_calibration_slope": calibration_slope(y_real, y_pred),
            "regressor_rmse_positive": rmse,
            "regressor_mae_positive": mae,
        }
    )
    metrics["regressor_draft_value_score"] = float(
        metrics["regressor_mean_ndcg_at_64"]
        - 0.25 * metrics["regressor_top64_weighted_mae_normalized"]
    )
    return metrics


def compute_cross_outcome_metrics(
    y_score: np.ndarray,
    outcome_columns: pl.DataFrame | None,
    draft_group: np.ndarray | None = None,
    draft_year: np.ndarray | None = None,
    meta_columns: pl.DataFrame | None = None,
    elite_cfg: ConfigLike = None,
    elite_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate one ranked board against multiple outcome families by draft class.

    Args:
        y_score: Model predictions used to rank players within each draft class.
        outcome_columns: Optional outcome labels used for the cross-outcome scorecard.
        draft_group: Preferred draft-class grouping key, such as `Universe:Year`.
        draft_year: Legacy alias for grouping when only the draft year is available.
        meta_columns: Metadata columns used by scoped elite definitions.
        elite_cfg: Config-like object describing derived elite label settings.
        elite_thresholds: Pre-fit elite thresholds, usually learned on the training split.

    Returns:
        Availability flags and cross-outcome metrics for each supported outcome family.
    """
    metrics: dict[str, float] = {
        "cross_outcomes_available": 0.0,
        "cross_econ_available": 0.0,
        "cross_talent_available": 0.0,
        "cross_longevity_available": 0.0,
        "cross_elite_available": 0.0,
        "cross_bust_available": 0.0,
    }
    if outcome_columns is None:
        return metrics

    group_key = draft_group if draft_group is not None else draft_year
    if group_key is None:
        group_key = np.full(y_score.shape, "all", dtype=object)

    def _first_available(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in outcome_columns.columns:
                return col
        return None

    def _topk_actual_value_by_group(y_val: np.ndarray, k: int) -> float:
        values: list[float] = []
        for group in np.unique(group_key):
            mask = group_key == group
            if not np.any(mask):
                continue
            scores_g = y_score[mask]
            y_g = y_val[mask]
            k_eff = min(k, y_g.size)
            order = np.argsort(-scores_g)[:k_eff]
            values.append(float(np.sum(y_g[order])))
        return float(np.mean(values)) if values else 0.0

    def _precision_at_k_by_group(y_binary: np.ndarray, k: int) -> float:
        vals: list[float] = []
        for group in np.unique(group_key):
            mask = group_key == group
            if not np.any(mask):
                continue
            scores_g = y_score[mask]
            y_g = y_binary[mask]
            k_eff = min(k, y_g.size)
            order = np.argsort(-scores_g)[:k_eff]
            vals.append(float(np.mean(y_g[order])))
        return float(np.mean(vals)) if vals else 0.0

    def _recall_at_k_by_group(y_binary: np.ndarray, k: int) -> float:
        vals: list[float] = []
        for group in np.unique(group_key):
            mask = group_key == group
            if not np.any(mask):
                continue
            scores_g = y_score[mask]
            y_g = y_binary[mask]
            positives = float(np.sum(y_g))
            if positives <= 0:
                vals.append(0.0)
                continue
            k_eff = min(k, y_g.size)
            order = np.argsort(-scores_g)[:k_eff]
            vals.append(float(np.sum(y_g[order]) / positives))
        return float(np.mean(vals)) if vals else 0.0

    def _derived_elite_labels() -> np.ndarray | None:
        if not bool(_cfg_get(elite_cfg, "enabled", False)):
            return None
        if outcome_columns is None:
            return None

        source_column = _cfg_get(elite_cfg, "source_column")
        if source_column is None or str(source_column) not in outcome_columns.columns:
            return None

        thresholds = elite_thresholds or fit_elite_thresholds(
            outcome_columns, meta_columns, elite_cfg
        )
        if not thresholds:
            return None

        y_source = np.maximum(outcome_columns[str(source_column)].to_numpy().astype(float), 0.0)
        scope = _cfg_get_str(elite_cfg, "scope", "global")
        scope_column = _cfg_get_str(elite_cfg, "scope_column", "Position_Group")
        global_threshold = float(thresholds.get("__global__", 0.0))

        if scope == "global" or meta_columns is None or scope_column not in meta_columns.columns:
            return (y_source >= global_threshold).astype(float)

        scope_values = meta_columns.get_column(scope_column).cast(pl.Utf8).to_numpy()
        labels = np.zeros(y_source.shape, dtype=float)
        for idx, scope_value in enumerate(scope_values):
            threshold = float(thresholds.get(str(scope_value), global_threshold))
            labels[idx] = 1.0 if y_source[idx] >= threshold else 0.0
        return labels

    def _bust_rate_at_k_by_group(y_success: np.ndarray, k: int) -> float:
        vals: list[float] = []
        for group in np.unique(group_key):
            mask = group_key == group
            if not np.any(mask):
                continue
            scores_g = y_score[mask]
            y_g = y_success[mask]
            k_eff = min(k, y_g.size)
            order = np.argsort(-scores_g)[:k_eff]
            vals.append(float(np.mean(1.0 - y_g[order])))
        return float(np.mean(vals)) if vals else 0.0

    metrics["cross_outcomes_available"] = 1.0

    econ_col = _first_available(
        ["Positive_Career_Merit_Cap_Share", "Career_Merit_Cap_Share", "Positive_DPO"]
    )
    if econ_col is not None:
        metrics["cross_econ_available"] = 1.0
        y_econ = outcome_columns[econ_col].to_numpy()
        metrics["cross_econ_mean_ndcg_at_64"] = mean_ndcg_by_group(y_econ, y_score, group_key, k=64)
        metrics["cross_econ_top64_actual_value"] = _topk_actual_value_by_group(y_econ, k=64)

    talent_col = _first_available(["Top3_Mean_Current_Overall", "Peak_Overall"])
    if talent_col is not None:
        metrics["cross_talent_available"] = 1.0
        y_talent = outcome_columns[talent_col].to_numpy()
        metrics["cross_talent_mean_ndcg_at_64"] = mean_ndcg_by_group(
            y_talent, y_score, group_key, k=64
        )

    longevity_col = _first_available(["Career_Games_Played"])
    if longevity_col is not None:
        metrics["cross_longevity_available"] = 1.0
        y_longevity = outcome_columns[longevity_col].to_numpy()
        metrics["cross_longevity_mean_ndcg_at_64"] = mean_ndcg_by_group(
            y_longevity, y_score, group_key, k=64
        )

    derived_elite = _derived_elite_labels()
    elite_col = None
    y_elite: np.ndarray | None = None
    if derived_elite is not None:
        y_elite = derived_elite
    else:
        elite_col = _first_available(
            ["Hall_of_Fame_Flag", "Hall_Of_Fame_Points", "HOF_Points", "Award_Count"]
        )
        if elite_col is not None:
            y_elite_raw = outcome_columns[elite_col].to_numpy()
            y_elite = (y_elite_raw > 0).astype(float)

    if y_elite is not None:
        metrics["cross_elite_available"] = 1.0
        precision_k = _cfg_get_int(elite_cfg, "top_k_precision", 32)
        recall_k = _cfg_get_int(elite_cfg, "top_k_recall", 64)
        metrics["cross_elite_precision_at_32"] = _precision_at_k_by_group(y_elite, k=precision_k)
        metrics["cross_elite_recall_at_64"] = _recall_at_k_by_group(y_elite, k=recall_k)

    success_col = _first_available(["Economic_Success", "Positive_Career_Merit_Cap_Share"])
    if success_col is not None:
        metrics["cross_bust_available"] = 1.0
        y_success_raw = outcome_columns[success_col].to_numpy()
        y_success = (y_success_raw > 0).astype(float)
        metrics["cross_bust_rate_at_32"] = _bust_rate_at_k_by_group(y_success, k=32)

    return metrics
