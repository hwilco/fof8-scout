import mlflow
import numpy as np
import polars as pl

from fof8_ml.models.registry import get_model_family
from fof8_ml.orchestration.evaluator import (
    compute_cross_outcome_metrics,
    fit_elite_thresholds,
    compute_regressor_oof_metrics,
    prefix_metrics,
    rename_metric_prefix,
)
from fof8_ml.orchestration.pipeline_runner import PipelineContext
from fof8_ml.orchestration.progress import (
    print_phase,
    print_refit_phase,
    print_test_phase,
    print_validation_run_start,
)
from fof8_ml.orchestration.trainer import (
    run_cv_regressor,
    train_final_model,
    train_model_with_validation,
)


def _concat_features(*frames: pl.DataFrame) -> pl.DataFrame:
    non_empty = [frame for frame in frames if len(frame) > 0]
    return pl.concat(non_empty, how="vertical_relaxed") if non_empty else frames[0].head(0)


def _concat_targets(*arrays: np.ndarray) -> np.ndarray:
    non_empty = [arr for arr in arrays if arr.size > 0]
    return np.concatenate(non_empty) if non_empty else np.array([], dtype=float)


def _empty_like_features(X: pl.DataFrame) -> pl.DataFrame:
    return X.head(0)


def _empty_meta() -> pl.DataFrame:
    return pl.DataFrame(
        {"Universe": [], "Year": [], "Position_Group": []},
        schema={"Universe": pl.String, "Year": pl.Int64, "Position_Group": pl.String},
    )


def _scope_meta_from_features(X: pl.DataFrame, elite_cfg: object | None) -> pl.DataFrame | None:
    scope_column = "Position_Group"
    if elite_cfg is not None and hasattr(elite_cfg, "get"):
        scope_column = str(elite_cfg.get("scope_column", scope_column))
    elif isinstance(elite_cfg, dict):
        scope_column = str(elite_cfg.get("scope_column", scope_column))

    if scope_column in X.columns:
        return X.select(pl.col(scope_column).cast(pl.Utf8))
    return None


def _rename_regressor_eval_metrics(
    metrics: dict[str, float], target_prefix: str
) -> dict[str, float]:
    renamed = rename_metric_prefix(metrics, "regressor_oof_", target_prefix)
    if "regressor_draft_value_score" in metrics:
        renamed[f"{target_prefix}draft_value_score"] = metrics["regressor_draft_value_score"]
        renamed.pop("regressor_draft_value_score", None)
    return renamed


def run_regressor(ctx: PipelineContext) -> dict[str, float]:
    cfg = ctx.cfg
    data = ctx.data
    quiet = ctx.sweep_context.quiet
    progress_every = int(cfg.get("runtime", {}).get("catboost_progress_every", 100))
    X_val = getattr(data, "X_val", _empty_like_features(data.X_train))
    X_test = getattr(data, "X_test", _empty_like_features(data.X_train))
    y_cls_val = getattr(data, "y_cls_val", np.array([], dtype=int))
    y_cls_test = getattr(data, "y_cls_test", np.array([], dtype=int))
    y_reg_val = getattr(data, "y_reg_val", np.array([], dtype=float))
    y_reg_test = getattr(data, "y_reg_test", np.array([], dtype=float))
    meta_val = getattr(data, "meta_val", _empty_meta())
    meta_test = getattr(data, "meta_test", _empty_meta())
    outcomes_val = getattr(data, "outcomes_val", None)
    outcomes_test = getattr(data, "outcomes_test", None)
    elite_cfg = cfg.target.get("outcome_scorecard", {}).get("elite")

    positive_mask = (data.y_cls == 1).astype(bool)
    X_reg = data.X_train.filter(pl.Series(positive_mask))
    required_meta_columns = {"Universe", "Year"}
    missing_meta_columns = sorted(required_meta_columns.difference(data.meta_train.columns))
    if missing_meta_columns:
        missing = ", ".join(missing_meta_columns)
        raise ValueError(
            "PreparedData.meta_train must include Universe and Year for draft-aware metrics. "
            f"Missing: {missing}."
        )
    meta_positive = data.meta_train.filter(pl.Series(positive_mask))
    outcomes_train_positive = (
        data.outcomes_train.filter(pl.Series(positive_mask))
        if data.outcomes_train is not None
        else None
    )
    scope_meta_positive = _scope_meta_from_features(X_reg, elite_cfg)
    elite_thresholds = fit_elite_thresholds(
        outcomes_train_positive, scope_meta_positive, elite_cfg
    )
    draft_group = (
        meta_positive.get_column("Universe").cast(pl.Utf8)
        + ":"
        + meta_positive.get_column("Year").cast(pl.Utf8)
    ).to_numpy()
    regressor_cfg = cfg.target.regressor_intensity
    target_space_value = (
        regressor_cfg.get("target_space", "log")
        if hasattr(regressor_cfg, "get")
        else getattr(regressor_cfg, "target_space", "log")
    )
    target_space = str(target_space_value).strip().lower()
    if target_space not in {"raw", "log"}:
        raise ValueError(
            f"Unsupported regressor target_space '{target_space}'. Expected 'raw' or 'log'."
        )

    model_family = get_model_family(role="regressor", model_name=cfg.model.name)
    loss_function = str(cfg.model.params.get("loss_function", ""))
    if model_family == "catboost" and loss_function.startswith("Tweedie") and target_space != "raw":
        raise ValueError(
            "CatBoost Tweedie requires target.regressor_intensity.target_space='raw'. "
            f"Got '{target_space}'."
        )

    y_reg_raw = data.y_reg[positive_mask]
    if model_family == "catboost" and loss_function.startswith("Tweedie") and np.any(y_reg_raw < 0):
        raise ValueError(
            "CatBoost Tweedie requires non-negative regressor targets, but negative values were "
            "found in the configured regressor target column."
        )
    y_reg_target = y_reg_raw if target_space == "raw" else np.log1p(y_reg_raw)

    if not quiet:
        n_hits = int(positive_mask.sum())
        print(f"\nFiltered to {n_hits} ground truth positive cases for regressor training.")
        print(f"Regressor target space: {target_space}")
        print("\n" + "=" * 40)
        print("REGRESSOR: INTENSITY REGRESSOR")
        print("=" * 40)

    with ctx.logger.start_model_run("regressor", ctx.sweep_context):
        ctx.logger.log_model_params(cfg.model, prefix="regressor")
        mlflow.log_param("target.regressor.target_col", cfg.target.regressor_intensity.target_col)
        mlflow.log_param("target.regressor.target_space", target_space)
        mlflow.log_param("regressor_target_space", target_space)
        val_positive_mask = (y_cls_val == 1).astype(bool)
        test_positive_mask = (y_cls_test == 1).astype(bool)
        has_validation = int(val_positive_mask.sum()) > 0
        mlflow.log_param("regressor_mode", "validation_holdout" if has_validation else "grouped_cv")

        if has_validation:
            if not quiet:
                print_validation_run_start(
                    role_label="regressor",
                    meta_train=meta_positive,
                    meta_val=meta_val.filter(pl.Series(val_positive_mask)),
                    meta_test=meta_test.filter(pl.Series(test_positive_mask)),
                    extra_detail=(
                        f"Positive-only training rows: train={len(X_reg)}, "
                        f"val={int(val_positive_mask.sum())}, "
                        f"test={int(test_positive_mask.sum())}."
                    ),
                )
                print_phase(
                    "Training regressor on positive cases with held-out universe validation"
                )
            X_val_reg = X_val.filter(pl.Series(val_positive_mask))
            y_val_reg_raw = y_reg_val[val_positive_mask]
            y_val_reg_target = y_val_reg_raw if target_space == "raw" else np.log1p(y_val_reg_raw)
            tuning_model = train_model_with_validation(
                model_cfg=cfg.model,
                role="regressor",
                X_train=X_reg,
                y_train=y_reg_target,
                X_val=X_val_reg,
                y_val=y_val_reg_target,
                seed=cfg.seed,
                interactive_progress_every=progress_every,
                use_gpu=cfg.use_gpu,
                quiet=quiet,
            )
            tuning_predictions = tuning_model.predict(X_val_reg)
            best_iterations = [tuning_model.get_best_iteration()]
            regressor_metrics = _rename_regressor_eval_metrics(
                compute_regressor_oof_metrics(
                    y_true=y_val_reg_target,
                    oof_predictions=tuning_predictions,
                    target_space=target_space,
                    draft_group=(
                        meta_val.filter(pl.Series(val_positive_mask))
                        .get_column("Universe")
                        .cast(pl.Utf8)
                        + ":"
                        + meta_val.filter(pl.Series(val_positive_mask))
                        .get_column("Year")
                        .cast(pl.Utf8)
                    ).to_numpy(),
                ),
                "regressor_val_",
            )
            val_score_raw = (
                np.expm1(tuning_predictions)
                if target_space == "log"
                else np.maximum(tuning_predictions, 0)
            )
            outcomes_val_positive = (
                outcomes_val.filter(pl.Series(val_positive_mask))
                if outcomes_val is not None
                else None
            )
            regressor_metrics.update(
                prefix_metrics(
                    compute_cross_outcome_metrics(
                        y_score=val_score_raw,
                        outcome_columns=outcomes_val_positive,
                        draft_group=(
                            meta_val.filter(pl.Series(val_positive_mask))
                            .get_column("Universe")
                            .cast(pl.Utf8)
                            + ":"
                            + meta_val.filter(pl.Series(val_positive_mask))
                            .get_column("Year")
                            .cast(pl.Utf8)
                        ).to_numpy(),
                        meta_columns=_scope_meta_from_features(X_val_reg, elite_cfg),
                        elite_cfg=elite_cfg,
                        elite_thresholds=elite_thresholds,
                    ),
                    "regressor_val_",
                )
            )
        else:
            groups = meta_positive.get_column("Universe").cast(pl.String).to_numpy()
            regressor_cv_result = run_cv_regressor(
                X=X_reg,
                y=y_reg_target,
                model_cfg=cfg.model,
                cv_cfg=cfg.cv,
                seed=cfg.seed,
                groups=groups,
                interactive_progress_every=progress_every,
                use_gpu=cfg.use_gpu,
                quiet=quiet,
                target_space=target_space,
            )

            regressor_metrics = compute_regressor_oof_metrics(
                y_true=y_reg_target,
                oof_predictions=regressor_cv_result.oof_predictions,
                target_space=target_space,
                draft_group=draft_group,
            )
            y_score_raw = (
                np.expm1(regressor_cv_result.oof_predictions)
                if target_space == "log"
                else np.maximum(regressor_cv_result.oof_predictions, 0)
            )
            regressor_metrics.update(
                compute_cross_outcome_metrics(
                    y_score=y_score_raw,
                    outcome_columns=outcomes_train_positive,
                    draft_group=draft_group,
                    meta_columns=scope_meta_positive,
                    elite_cfg=elite_cfg,
                    elite_thresholds=elite_thresholds,
                )
            )

            cv_rmse = [m["rmse"] for m in regressor_cv_result.fold_metrics]
            cv_mae = [m["mae"] for m in regressor_cv_result.fold_metrics]
            regressor_metrics["regressor_mean_rmse"] = float(np.mean(cv_rmse))
            regressor_metrics["regressor_mean_mae"] = float(np.mean(cv_mae))
            best_iterations = regressor_cv_result.best_iterations

        avg_best_iters = int(np.mean(best_iterations)) if best_iterations else 100
        final_train_y_raw = (
            _concat_targets(data.y_reg[positive_mask], y_reg_val[val_positive_mask])
            if has_validation
            else y_reg_raw
        )
        if has_validation:
            final_train_X = _concat_features(
                data.X_train.filter(pl.Series(positive_mask)),
                X_val.filter(pl.Series(val_positive_mask)),
            )
            final_train_y_raw = _concat_targets(
                data.y_reg[positive_mask], y_reg_val[val_positive_mask]
            )
        else:
            final_train_X = X_reg
        final_train_y = final_train_y_raw if target_space == "raw" else np.log1p(final_train_y_raw)
        if has_validation and not quiet:
            print_refit_phase("regressor", len(X_reg), int(val_positive_mask.sum()))
        regressor_model = train_final_model(
            model_cfg=cfg.model,
            role="regressor",
            X=final_train_X,
            y=final_train_y,
            avg_best_iterations=avg_best_iters,
            seed=cfg.seed,
            interactive_progress_every=progress_every,
            use_gpu=cfg.use_gpu,
            quiet=quiet,
        )

        X_test_reg = X_test.filter(pl.Series(test_positive_mask))
        if len(X_test_reg) > 0:
            if not quiet:
                print_test_phase("regressor", len(X_test_reg))
            y_test_reg_raw = y_reg_test[test_positive_mask]
            y_test_reg_target = (
                y_test_reg_raw if target_space == "raw" else np.log1p(y_test_reg_raw)
            )
            test_predictions = regressor_model.predict(X_test_reg)
            test_draft_group = (
                meta_test.filter(pl.Series(test_positive_mask)).get_column("Universe").cast(pl.Utf8)
                + ":"
                + meta_test.filter(pl.Series(test_positive_mask)).get_column("Year").cast(pl.Utf8)
            ).to_numpy()
            test_metrics = _rename_regressor_eval_metrics(
                compute_regressor_oof_metrics(
                    y_true=y_test_reg_target,
                    oof_predictions=test_predictions,
                    target_space=target_space,
                    draft_group=test_draft_group,
                ),
                "regressor_test_",
            )
            outcomes_test_positive = (
                outcomes_test.filter(pl.Series(test_positive_mask))
                if outcomes_test is not None
                else None
            )
            test_score_raw = (
                np.expm1(test_predictions)
                if target_space == "log"
                else np.maximum(test_predictions, 0)
            )
            test_metrics.update(
                prefix_metrics(
                    compute_cross_outcome_metrics(
                        y_score=test_score_raw,
                        outcome_columns=outcomes_test_positive,
                        draft_group=test_draft_group,
                        meta_columns=_scope_meta_from_features(X_test_reg, elite_cfg),
                        elite_cfg=elite_cfg,
                        elite_thresholds=elite_thresholds,
                    ),
                    "regressor_test_",
                )
            )
        else:
            test_metrics = {}
            test_score_raw = None

        ctx.logger.log_regressor_results(
            regressor_metrics,
            regressor_model,
            final_train_X,
            cfg,
            quiet,
            test_predictions=test_score_raw,
            test_meta=meta_test.filter(pl.Series(test_positive_mask))
            if len(meta_test)
            else meta_test,
            test_metrics=test_metrics,
        )

        mlflow.log_metrics(
            {
                **regressor_metrics,
                **test_metrics,
            }
        )

    draft_value_key = (
        "regressor_val_draft_value_score" if has_validation else "regressor_draft_value_score"
    )
    rmse_key = "regressor_val_rmse" if has_validation else "regressor_oof_rmse"
    mae_key = "regressor_val_mae" if has_validation else "regressor_oof_mae"
    return {
        rmse_key: regressor_metrics[rmse_key],
        mae_key: regressor_metrics[mae_key],
        draft_value_key: regressor_metrics[draft_value_key],
        "regressor_test_draft_value_score": test_metrics.get(
            "regressor_test_draft_value_score", 0.0
        ),
    }
