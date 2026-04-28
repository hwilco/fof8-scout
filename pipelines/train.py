import logging
import warnings
import contextlib

# Suppress Optuna deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.distributions")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="hydra_plugins.hydra_optuna_sweeper"
)
# Suppress Hydra experimental feature warnings
warnings.filterwarnings("ignore", message=".*multivariate.*experimental feature.*")

import fnmatch
import hydra
import mlflow
import mlflow.data
import dagshub
import dvc.api
import polars as pl
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    recall_score,
    precision_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
)
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from fof8_core.loader import FOF8Loader
from fof8_core.features import apply_position_mask_ml_v1 as apply_position_mask
import subprocess
import json
from fof8_ml.evaluation.metrics import calculate_survival_metrics
from fof8_ml.evaluation.plotting import log_feature_importance, log_confusion_matrix

# Dynamic Model Loading
from fof8_ml.models import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
    SklearnRegressorWrapper,
)

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@contextlib.contextmanager
def preserve_cwd(new_cwd: str = None):
    """Temporarily change the working directory."""
    old_cwd = os.getcwd()
    if new_cwd:
        os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def get_model_wrapper(
    model_name: str,
    stage: str,
    random_seed: int,
    params: dict,
    use_gpu: bool = False,
    thread_count: int = -1,
):
    model_name = model_name.lower()
    if stage == "stage1":
        if "catboost" in model_name:
            return CatBoostClassifierWrapper(
                random_seed=random_seed, use_gpu=use_gpu, thread_count=thread_count, **params
            )
        elif "xgb" in model_name:
            return XGBoostClassifierWrapper(random_seed=random_seed, use_gpu=use_gpu, **params)
        else:
            raise ValueError(f"Unknown model for stage 1: {model_name}")
    elif stage == "stage2":
        if "catboost" in model_name:
            return CatBoostRegressorWrapper(
                random_seed=random_seed, use_gpu=use_gpu, thread_count=thread_count, **params
            )
        elif "xgb" in model_name:
            return XGBoostRegressorWrapper(random_seed=random_seed, use_gpu=use_gpu, **params)
        elif "sklearn" in model_name or "tweedie" in model_name or "gamma" in model_name:
            return SklearnRegressorWrapper(model_name=model_name, use_gpu=use_gpu, **params)
        else:
            raise ValueError(f"Unknown model for stage 2: {model_name}")


@hydra.main(version_base=None, config_path="conf", config_name="economic_pipeline")
def main(cfg: DictConfig):
    # Initialize DagsHub tracking
    # This automatically sets MLFLOW_TRACKING_URI and credentials
    dagshub.init(repo_owner="hwilco", repo_name="fof8-scout", mlflow=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_root = os.path.abspath(os.path.join(script_dir, ".."))
    # db_path = os.path.join(exp_root, "mlflow.db")
    # artifact_root = os.path.join(exp_root, "mlruns")

    # Universal autologging for CatBoost, XGBoost, Sklearn
    mlflow.autolog(log_models=False)  # We log models manually for better control

    # Use local SQLite only if DagsHub/Remote URI is not set
    # if not os.environ.get("MLFLOW_TRACKING_URI"):
    #     mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(cfg.experiment_name)
    if exp is None:
        try:
            client.create_experiment(cfg.experiment_name)
        except Exception:
            # Handle race condition where experiment is created by another process
            pass

    exp = mlflow.set_experiment(cfg.experiment_name)

    # Detect Sweep / Multirun context
    is_sweep = False
    try:
        is_sweep = HydraConfig.get().mode == RunMode.MULTIRUN
    except:
        pass

    sweep_run_id = None
    sweep_name = None
    if is_sweep or cfg.get("sweep_name"):
        # Determine sweep name
        if cfg.get("sweep_name"):
            sweep_name = cfg.sweep_name
        else:
            # Fallback to date-based name if in multirun
            sweep_name = f"Sweep_{os.path.basename(HydraConfig.get().sweep.dir)}"

        # Get/Create Parent Run (Idempotent for parallel trials)
        existing_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{sweep_name}'",
            max_results=1,
        )
        if existing_runs:
            sweep_run_id = existing_runs[0].info.run_id
        else:
            try:
                with mlflow.start_run(run_name=sweep_name) as sweep_parent:
                    sweep_run_id = sweep_parent.info.run_id
                    # Log Search Space to Parent for visibility
                    if (
                        "hydra" in cfg
                        and "sweeper" in cfg.hydra
                        and "search_space" in cfg.hydra.sweeper
                    ):
                        search_space = OmegaConf.to_container(
                            cfg.hydra.sweeper.search_space, resolve=True
                        )
                        mlflow.log_params(flatten_dict(search_space, parent_key="search_space"))
            except Exception:
                # Handle race condition in parallel starts
                existing_runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{sweep_name}'",
                    max_results=1,
                )
                if existing_runs:
                    sweep_run_id = existing_runs[0].info.run_id

    # Resolve paths relative to experiment root to allow running from any directory
    absolute_raw_path = os.path.abspath(os.path.join(exp_root, cfg.data.raw_path))
    loader = FOF8Loader(base_path=absolute_raw_path, league_name=cfg.data.league_name)

    # 1. Automate Timeline Discovery
    initial_year = loader.initial_sim_year
    final_sim_year = loader.final_sim_year
    valid_start_year = initial_year + 1

    # 2. Load Preprocessed Data
    features_file = os.path.abspath(os.path.join(exp_root, cfg.data.features_path))

    if not os.path.exists(features_file):
        raise FileNotFoundError(
            f"Processed features not found at {features_file}. Run transform.py first."
        )

    df = pl.read_parquet(features_file)

    # --- Runtime Filtering & Target Labeling ---
    # 1. Apply Merit Threshold to define the binary target
    df = df.with_columns(
        (pl.col("Career_Merit_Cap_Share") > cfg.target.stage1_sieve.merit_threshold)
        .alias("Cleared_Sieve")
        .cast(pl.Int8)
    )

    # 2. Filter by Positions
    if cfg.positions and cfg.positions != "all":
        pos_list = [cfg.positions] if isinstance(cfg.positions, str) else cfg.positions
        df = df.filter(pl.col("Position_Group").is_in(pos_list))

    # 3. Apply Right Censor Buffer (Years where players haven't finished careers)
    valid_end_year = final_sim_year - cfg.split.right_censor_buffer
    df = df.filter(pl.col("Year") <= valid_end_year)

    # --- In-Memory Chronological Split ---
    total_valid_years = valid_end_year - valid_start_year + 1
    test_years_count = int(total_valid_years * cfg.split.test_split_pct)
    train_end_year = valid_end_year - test_years_count

    train_year_range = [valid_start_year, train_end_year]
    test_year_range = [train_end_year + 1, valid_end_year]

    if not (is_sweep and cfg.get("quiet_sweep", False)):
        print(f"Simulation Range: {initial_year} to {final_sim_year}")
        print(
            f"Active Range: {valid_start_year} to {valid_end_year} (Buffer: {cfg.split.right_censor_buffer} years)"
        )
        print(
            f"Training Set: {train_year_range} ({train_year_range[1] - train_year_range[0] + 1} draft classes)"
        )
        print(
            f"Holdout Set: {test_year_range} ({test_year_range[1] - test_year_range[0] + 1} draft classes)"
        )

    train_df = df.filter(
        (pl.col("Year") >= train_year_range[0]) & (pl.col("Year") <= train_year_range[1])
    )
    test_df = df.filter(
        (pl.col("Year") >= test_year_range[0]) & (pl.col("Year") <= test_year_range[1])
    )

    metadata_cols = ["Player_ID", "Year", "First_Name", "Last_Name"]
    # Phase 1 & 2: Dynamic Target & Leakage Prevention
    # We strip out both our learning targets and any manually defined leakage columns
    target_cols = [
        cfg.target.stage1_sieve.target_col,
        cfg.target.stage2_intensity.target_col
    ] + list(cfg.target.leakage_prevention.drop_cols)

    feature_cols = [c for c in df.columns if c not in metadata_cols and c not in target_cols]

    X_train = train_df.select(feature_cols)
    y_train_df = train_df.select(target_cols)
    meta_train = train_df.select(metadata_cols)

    X_test = test_df.select(feature_cols)
    y_test_df = test_df.select(target_cols)
    meta_test = test_df.select(metadata_cols)

    if cfg.get("mask_positional_features", False):
        if not (is_sweep and cfg.get("quiet_sweep", False)):
            print("Applying In-Memory Positional Feature Masking...")
        X_train = apply_position_mask(X_train)
        X_test = apply_position_mask(X_test)

    include_features = cfg.get("include_features")
    if include_features:
        # Expand wildcards for inclusion
        all_cols = X_train.columns
        expanded_include = []
        for p in include_features:
            if "*" in p or "?" in p:
                expanded_include.extend(fnmatch.filter(all_cols, p))
            else:
                expanded_include.append(p)

        # Unique features that actually exist
        include_cols = [c for c in list(dict.fromkeys(expanded_include)) if c in all_cols]

        print(
            f"Applying Feature Ablation: Keeping {len(include_cols)} features matching {include_features}"
        )
        X_train = X_train.select(include_cols)
        X_test = X_test.select(include_cols)

    exclude_features = cfg.get("exclude_features")
    if exclude_features:
        # Expand wildcards for exclusion
        all_cols = X_train.columns
        expanded_exclude = []
        for p in exclude_features:
            if "*" in p or "?" in p:
                expanded_exclude.extend(fnmatch.filter(all_cols, p))
            else:
                expanded_exclude.append(p)

        cols_to_drop = [c for c in list(dict.fromkeys(expanded_exclude)) if c in all_cols]
        if cols_to_drop:
            print(
                f"Applying Feature Ablation: Excluding {len(cols_to_drop)} features matching {exclude_features}"
            )
            X_train = X_train.drop(cols_to_drop)
            X_test = X_test.drop(cols_to_drop)

    y_cls = y_train_df.get_column(cfg.target.stage1_sieve.target_col).to_numpy()
    y_reg = y_train_df.get_column(cfg.target.stage2_intensity.target_col).to_numpy()

    # If in a sweep, we link to the sweep_run_id via tags
    tags = {}
    if sweep_run_id:
        tags["mlflow.parentRunId"] = sweep_run_id
        tags["sweep_run_id"] = sweep_run_id
    if sweep_name:
        tags["sweep_name"] = sweep_name

    if is_sweep:
        trial_num = HydraConfig.get().job.num + 1
        n_trials = HydraConfig.get().sweeper.n_trials
        print("\n" + ">" * 10 + f" STARTING TRIAL {trial_num}/{n_trials} " + "<" * 10)

    with mlflow.start_run(
        run_name=f"Pipeline_{cfg.stage1_model.name}_{cfg.stage2_model.name}", tags=tags
    ) as pipeline_run:
        # Log DagsHub and dataset metadata
        mlflow.set_tag("data.league", cfg.data.league_name)
        mlflow.set_tag("data.raw_path", cfg.data.raw_path)

        # Log Git Commit and DVC Data Version
        try:
            repo_root = os.path.abspath(os.path.join(exp_root, ".."))
            relative_data_path = os.path.relpath(absolute_raw_path, repo_root)
            with preserve_cwd(repo_root):
                git_commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                )
                mlflow.set_tag("git_commit", git_commit)

                try:
                    data_url = dvc.api.get_url(path=relative_data_path, remote="origin")
                    mlflow.set_tag("dvc.data_url", data_url)
                except:
                    pass
        except Exception as e:
            logging.warning(f"Could not log Git commit / DVC version: {e}")
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(flatten_dict(cfg_container))

        if "tags" in cfg and cfg.tags:
            mlflow.set_tags(OmegaConf.to_container(cfg.tags, resolve=True))

        if is_sweep:
            mlflow.set_tag("trial_num", str(HydraConfig.get().job.num))

        n_pos = int(y_cls.sum())
        n_neg = len(y_cls) - n_pos
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        if not (is_sweep and cfg.get("quiet_sweep", False)):
            print(
                f"Stage 1 Class Balance - Positive Merit (Hits): {n_pos}, Negative Merit (Busts): {n_neg}, Ratio: {pos_weight:.2f}"
            )
        mlflow.log_params({"data.n_pos": n_pos, "data.n_neg": n_neg})

        stage1_run_id = cfg.get("stage1_run_id")
        best_f1_0 = 0.0
        hit_recall = 0.0
        best_threshold = 0.5
        s1_oof_pr_auc = 0.0
        s2_oof_rmse = 0.0

        # ---------------------------------------------------------
        # STAGE 1: SIEVE CLASSIFIER
        # ---------------------------------------------------------
        if not stage1_run_id:
            with mlflow.start_run(run_name="Stage1_Sieve_Classifier", nested=True) as stage1_run:
                mlflow.set_tag("model_stage", "stage1")
                if sweep_name:
                    mlflow.set_tag("sweep_name", sweep_name)
                if sweep_run_id:
                    mlflow.set_tag("sweep_run_id", sweep_run_id)

                # Log Stage 1 specific parameters for easier comparison
                s1_params = OmegaConf.to_container(cfg.stage1_model.params, resolve=True)
                mlflow.log_params(flatten_dict(s1_params, parent_key="s1"))
                if not (is_sweep and cfg.get("quiet_sweep", False)):
                    print("\n" + "=" * 40)
                    print("STAGE 1: SIEVE CLASSIFIER")
                    print("=" * 40)

                skf = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=cfg.cv.shuffle, random_state=cfg.seed)
                indices = np.arange(len(X_train))
                oof_probs = np.zeros(len(X_train))

                cv_metrics = []
                best_iterations = []

                for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_cls)):
                    if not (is_sweep and cfg.get("quiet_sweep", False)):
                        print(f"--- S1 Fold {fold} ---")
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_cls[train_idx], y_cls[val_idx]

                    params = OmegaConf.to_container(cfg.stage1_model.params, resolve=True)
                    use_gpu = cfg.get("use_gpu", False)
                    thread_count = cfg.stage1_model.params.get("thread_count", -1)
                    model = get_model_wrapper(
                        cfg.stage1_model.name,
                        "stage1",
                        cfg.seed,
                        params,
                        use_gpu=use_gpu,
                        thread_count=thread_count,
                    )

                    model.fit(X_cv_train, y_cv_train, X_cv_val, y_cv_val)
                    best_iterations.append(model.get_best_iteration())
                    y_val_prob = model.predict_proba(X_cv_val)

                    oof_probs[val_idx] = y_val_prob

                    metrics = calculate_survival_metrics(y_cv_val, y_val_prob)
                    cv_metrics.append(metrics)
                    for m_name, m_val in metrics.items():
                        mlflow.log_metric(f"fold_{fold}_{m_name}", m_val)

                summary_metrics = {}
                for key in cv_metrics[0].keys():
                    values = [m[key] for m in cv_metrics]
                    summary_metrics[f"mean_{key}"] = np.mean(values)
                mlflow.log_metrics(summary_metrics)

                # Threshold Optimization
                if not (is_sweep and cfg.get("quiet_sweep", False)):
                    print(
                        f"\nOptimizing Stage 1 Threshold (Constraint: Min Survivor Recall >= {cfg.target.stage1_sieve.min_survivor_recall})..."
                    )
                best_threshold = 0.5
                best_f1_0 = -1.0
                y_true = y_cls

                thresholds = np.linspace(0.01, 0.99, 99)
                for thresh in thresholds:
                    current_preds = (oof_probs >= thresh).astype(int)
                    f1_0 = f1_score(y_true, current_preds, pos_label=0)
                    recall_1 = recall_score(y_true, current_preds)

                    if recall_1 >= cfg.target.stage1_sieve.min_survivor_recall:
                        if f1_0 > best_f1_0:
                            best_f1_0 = f1_0
                            best_threshold = thresh

                if best_f1_0 == -1.0:
                    best_threshold = thresholds[0]
                    final_preds = (oof_probs >= best_threshold).astype(int)
                    best_f1_0 = f1_score(y_true, final_preds, pos_label=0)
                else:
                    final_preds = (oof_probs >= best_threshold).astype(int)

                busts_filtered = np.sum((y_true == 0) & (final_preds == 0))
                hit_recall = recall_score(y_true, final_preds)
                bust_precision = precision_score(y_true, final_preds, pos_label=0)
                bust_recall = recall_score(y_true, final_preds, pos_label=0)

                p, r, _ = precision_recall_curve(y_true, oof_probs)
                s1_oof_pr_auc = auc(r, p)
                s1_oof_roc_auc = roc_auc_score(y_true, oof_probs)

                mlflow.log_params({"s1_optimal_threshold": best_threshold})
                mlflow.log_metrics(
                    {
                        "s1_oof_busts_filtered": busts_filtered,
                        "s1_oof_hit_recall": hit_recall,
                        "s1_oof_f1_bust": best_f1_0,
                        "s1_oof_precision_bust": bust_precision,
                        "s1_oof_recall_bust": bust_recall,
                        "s1_oof_pr_auc": s1_oof_pr_auc,
                        "s1_oof_roc_auc": s1_oof_roc_auc,
                    }
                )

                log_confusion_matrix(y_cls, final_preds, best_threshold)

                oof_df = meta_train.with_columns(
                    [
                        pl.Series("y_true", y_cls),
                        pl.Series("oof_prob", oof_probs),
                        pl.Series("cleared_sieve", final_preds),
                    ]
                )
                oof_df.write_csv("stage1_oof_results.csv")
                mlflow.log_artifact("stage1_oof_results.csv")

                # Train Final Stage 1 Model
                avg_best_iters = int(np.mean(best_iterations))
                final_params_s1 = OmegaConf.to_container(cfg.stage1_model.params, resolve=True)
                final_params_s1.pop("early_stopping_rounds", None)

                if "catboost" in cfg.stage1_model.name.lower():
                    final_params_s1["iterations"] = avg_best_iters
                elif "xgb" in cfg.stage1_model.name.lower():
                    final_params_s1["n_estimators"] = avg_best_iters

                stage1_model = get_model_wrapper(
                    cfg.stage1_model.name,
                    "stage1",
                    cfg.seed,
                    final_params_s1,
                    use_gpu=cfg.get("use_gpu", False),
                    thread_count=cfg.stage1_model.params.get("thread_count", -1),
                )
                stage1_model.fit(X_train, y_cls)
                stage1_model.log_model("stage1_model")

                if cfg.diagnostics.log_importance or cfg.diagnostics.log_shap:
                    is_cb = "catboost" in cfg.stage1_model.name.lower()
                    log_feature_importance(
                        stage1_model.model,
                        X_train.columns,
                        "Stage 1 Importance",
                        is_cb,
                        X=X_train.to_pandas(),
                        log_shap=cfg.diagnostics.log_shap,
                    )

            mlflow.log_metrics(
                {
                    "s1_oof_f1_bust": best_f1_0,
                    "s1_oof_hit_recall": hit_recall,
                    "s1_optimal_threshold": best_threshold,
                    "s1_oof_pr_auc": s1_oof_pr_auc,
                }
            )

            mask = (y_cls == 1).astype(bool)
        else:
            print(f"\nSKIPPING STAGE 1: Using results from Run {stage1_run_id}")
            oof_path = client.download_artifacts(stage1_run_id, "stage1_oof_results.csv")
            oof_df = pl.read_csv(oof_path)
            mask = (y_cls == 1).astype(bool)

            mlflow.set_tag("stage1_source_run", stage1_run_id)
            try:
                best_f1_0 = client.get_run(stage1_run_id).data.metrics.get("s1_oof_f1_bust", 0.0)
            except:
                best_f1_0 = 0.0

        # ---------------------------------------------------------
        # STAGE 2: INTENSITY REGRESSOR
        # ---------------------------------------------------------
        if cfg.get("train_stage2", True):
            with mlflow.start_run(run_name="Stage2_Intensity_Regressor", nested=True) as stage2_run:
                mlflow.set_tag("model_stage", "stage2")
                if sweep_name:
                    mlflow.set_tag("sweep_name", sweep_name)
                if sweep_run_id:
                    mlflow.set_tag("sweep_run_id", sweep_run_id)

                # Log Stage 2 specific parameters for easier comparison
                s2_params = OmegaConf.to_container(cfg.stage2_model.params, resolve=True)
                mlflow.log_params(flatten_dict(s2_params, parent_key="s2"))
                if not (is_sweep and cfg.get("quiet_sweep", False)):
                    print("\n" + "=" * 40)
                    print("STAGE 2: INTENSITY REGRESSOR")
                    print("=" * 40)

                X_reg = X_train.filter(pl.Series(mask))
                y_reg_target = np.log1p(y_reg[mask])

                kf = KFold(n_splits=cfg.cv.n_folds, shuffle=cfg.cv.shuffle, random_state=cfg.seed)
                indices_reg = np.arange(len(X_reg))
                oof_preds_reg = np.zeros(len(X_reg))

                cv_rmse = []
                cv_mae = []
                best_iters_reg = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(indices_reg)):
                    if not (is_sweep and cfg.get("quiet_sweep", False)):
                        print(f"--- S2 Fold {fold} ---")
                    X_cv_train, X_cv_val = X_reg[train_idx], X_reg[val_idx]
                    y_cv_train, y_cv_val = y_reg_target[train_idx], y_reg_target[val_idx]

                    params = OmegaConf.to_container(cfg.stage2_model.params, resolve=True)
                    use_gpu = cfg.get("use_gpu", False)
                    thread_count = cfg.stage2_model.params.get("thread_count", -1)
                    model = get_model_wrapper(
                        cfg.stage2_model.name,
                        "stage2",
                        cfg.seed,
                        params,
                        use_gpu=use_gpu,
                        thread_count=thread_count,
                    )

                    model.fit(X_cv_train, y_cv_train, X_cv_val, y_cv_val)
                    best_iters_reg.append(model.get_best_iteration())
                    y_val_pred = model.predict(X_cv_val)

                    oof_preds_reg[val_idx] = y_val_pred

                    y_val_real = np.expm1(y_cv_val)
                    y_val_pred_real = np.expm1(y_val_pred)

                    rmse = np.sqrt(mean_squared_error(y_val_real, y_val_pred_real))
                    mae = mean_absolute_error(y_val_real, y_val_pred_real)

                    cv_rmse.append(rmse)
                    cv_mae.append(mae)

                    mlflow.log_metric(f"fold_{fold}_rmse", rmse)
                    mlflow.log_metric(f"fold_{fold}_mae", mae)

                s2_oof_rmse = np.sqrt(
                    mean_squared_error(np.expm1(y_reg_target), np.expm1(oof_preds_reg))
                )
                s2_oof_mae = mean_absolute_error(np.expm1(y_reg_target), np.expm1(oof_preds_reg))

                mlflow.log_metrics(
                    {
                        "s2_oof_rmse": s2_oof_rmse,
                        "s2_oof_mae": s2_oof_mae,
                        "s2_mean_rmse": np.mean(cv_rmse),
                        "s2_mean_mae": np.mean(cv_mae),
                    }
                )

                mlflow.log_metrics({"s2_oof_rmse": s2_oof_rmse})

                # Train Final Stage 2 Model
                avg_best_iters_reg = int(np.mean(best_iters_reg)) if best_iters_reg else 100
                final_params_s2 = OmegaConf.to_container(cfg.stage2_model.params, resolve=True)
                final_params_s2.pop("early_stopping_rounds", None)

                if "catboost" in cfg.stage2_model.name.lower():
                    final_params_s2["iterations"] = avg_best_iters_reg
                elif "xgb" in cfg.stage2_model.name.lower():
                    final_params_s2["n_estimators"] = avg_best_iters_reg

                stage2_model = get_model_wrapper(
                    cfg.stage2_model.name,
                    "stage2",
                    cfg.seed,
                    final_params_s2,
                    use_gpu=cfg.get("use_gpu", False),
                    thread_count=cfg.stage2_model.params.get("thread_count", -1),
                )
                stage2_model.fit(X_reg, y_reg_target)
                stage2_model.log_model("stage2_model")

                if cfg.diagnostics.log_importance or cfg.diagnostics.log_shap:
                    is_cb = "catboost" in cfg.stage2_model.name.lower()
                    log_feature_importance(
                        stage2_model.model,
                        X_reg.columns,
                        "Stage 2 Importance",
                        is_cb,
                        X=X_reg.to_pandas(),
                        log_shap=cfg.diagnostics.log_shap,
                    )

        print("\nFull Pipeline Training Complete. Models saved to MLflow.")

        # --- Metric Consolidation for Optimization/Return ---
        available_metrics = {
            "s1_oof_f1_bust": best_f1_0,
            "s1_oof_recall_bust": bust_recall if "bust_recall" in locals() else 0.0,
            "s1_oof_pr_auc": s1_oof_pr_auc,
            "s2_oof_rmse": s2_oof_rmse,
        }

        # Fetch the target metric from the Hydra config
        opt_metric = cfg.get("optimization", {}).get(
            "metric", "s2_oof_rmse" if cfg.get("train_stage2", True) else "s1_oof_pr_auc"
        )
        higher_is_better = cfg.get("optimization", {}).get("direction", "maximize") == "maximize"

        current_score = available_metrics.get(opt_metric)
        if current_score is None:
            raise ValueError(
                f"Metric '{opt_metric}' is not available for optimization. Available metrics: {list(available_metrics.keys())}"
            )

        score_name = f"best_{opt_metric}"
        score_label = score_name.replace("best_", "").upper()

        # --- Update Sweep Parent with Best Results ---
        if sweep_run_id:
            # Compare against the parent champion
            parent_run_data = client.get_run(sweep_run_id).data
            previous_best = parent_run_data.metrics.get(score_name)

            is_new_best = False
            if previous_best is None:
                is_new_best = True
            elif higher_is_better and current_score > previous_best:
                is_new_best = True
            elif not higher_is_better and current_score < previous_best:
                is_new_best = True

            if is_new_best:
                previous_champion_id = parent_run_data.tags.get("best_trial_id")
                trial_num = HydraConfig.get().job.num

                # 1. Update Parent Metrics/Tags
                client.log_metric(sweep_run_id, score_name, current_score)
                client.set_tag(sweep_run_id, "best_trial_id", pipeline_run.info.run_id)

                # 2. Log a Note (Markdown) on the Parent with a link
                # Note: This uses a relative link that works within the MLflow UI
                note_content = f"### 🏆 Current Sweep Champion\n- **Run ID:** `{pipeline_run.info.run_id}`\n- **{score_label}:** {current_score:.4f}\n- **Params:** {str({k: v for k, v in cfg_container.items() if k in ['stage1_model', 'stage2_model']})}"
                client.set_tag(sweep_run_id, "mlflow.note.content", note_content)

                # 3. "Bubble Up" the Champion Tag on the child run
                if previous_champion_id:
                    try:
                        client.delete_tag(previous_champion_id, "champion")
                    except:
                        pass  # Might fail if the run was deleted or tag missing
                client.set_tag(pipeline_run.info.run_id, "champion", "true")

                # Log trial params as a clean summary
                trial_params = {
                    k: v for k, v in cfg_container.items() if k in ["stage1_model", "stage2_model"]
                }
                client.set_tag(sweep_run_id, "best_params", str(trial_params))

                # Register the model to the DagsHub Model Registry
                if cfg.get("train_stage2", True):
                    mlflow.register_model(
                        model_uri=f"runs:/{pipeline_run.info.run_id}/stage2_model",
                        name="fof8-scout-regressor",
                    )

            # --- Live Leaderboard Dashboard ---
            trial_num = HydraConfig.get().job.num + 1
            n_trials = HydraConfig.get().sweeper.n_trials

            # Refresh champion data from parent for the dashboard
            parent_run = client.get_run(sweep_run_id)
            best_score = parent_run.data.metrics.get(score_name)
            best_trial_id = parent_run.data.tags.get("best_trial_id")

            best_trial_num = "?"
            if best_trial_id:
                try:
                    best_trial_run = client.get_run(best_trial_id)
                    best_trial_num = best_trial_run.data.tags.get("trial_num", "?")
                except:
                    pass

            # Dynamic Label for display: Convert 'best_s1_f1_bust' -> 'S1_F1_BUST'
            score_label = score_name.replace("best_", "").upper()

            print("\n" + "=" * 60)
            print(f"🏆 SWEEP LEADERBOARD | Trial [{trial_num}/{n_trials}]")
            print("-" * 60)
            print(
                f"LATEST TRIAL RESULT: {current_score:.4f} ({score_label}) ({'IMPROVEMENT' if is_new_best else 'No improvement'})"
            )
            print(f"BEST SO FAR:         {best_score:.4f} ({score_label}) [Trial {best_trial_num}]")
            if is_new_best:
                print(f"NEW CHAMPION PARAMS: {trial_params}")
            print("=" * 60 + "\n")

        # Write lightweight outputs/metrics.json to satisfy DVC
        out_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "outputs"))
        os.makedirs(out_dir, exist_ok=True)
        metrics_file = os.path.join(out_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump({opt_metric: current_score}, f)

        return current_score


if __name__ == "__main__":
    main()
