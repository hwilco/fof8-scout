# FOF8-Scout: Architecture Refactoring Plan

## 1. Context & Objectives
This project is an ML experimentation pipeline (`fof8-experimentation`) built on top of a simulation data parser (`fof8-core`). The goal of this refactor is to eliminate data leakage, decouple physical data materialization from the ML training loop, and implement a strictly typed, modular Hydra configuration taxonomy.

**Strict Agent Directives:**
* **Data Processing:** Maintain strict use of `polars` for all data transformations. Do not regress to `pandas` unless strictly required by a downstream ML library (e.g., scikit-learn).
* **Environment & Tooling:** Environment management and dependency resolution are handled via `uv`. Code must remain compliant with modern formatting and linting standards via `ruff`.
* **Immutability:** `fof8-core` must remain ML-agnostic. It handles raw extraction and metric logic only. Do not add ML-specific code to the core library.

---

## Phase 1: Fix Target Leakage in Pipeline
**Goal:** Prevent the model from seeing future outcomes during training.

* **Target File:** `src/fof8_ml/train_pipeline.py`
* **The Bug:** `Career_Merit_Cap_Share` is physically present in `features.parquet` and is being accidentally included in the dynamic `feature_cols` list.
* **The Fix:** Explicitly add `"Career_Merit_Cap_Share"` to the `target_cols` list so it is stripped from `X_train` and `X_test` before model fitting. This acts as a temporary safety net until Phase 3 is complete.

---

## Phase 2: Hydra Configuration Restructuring
**Goal:** Eliminate the monolithic god-config and separate concerns into interchangeable components using a strict taxonomy. Do not use ambiguous keys like `name` or `binary_target`.

1. **Create Component Configs:**
   * **`conf/target/economic.yaml`**: Must strictly adhere to this schema:
     ```yaml
     stage1_sieve:
       target_col: "Cleared_Sieve"
       merit_threshold: 0
       min_survivor_recall: 0.95
     stage2_intensity:
       target_col: "DGO"
     leakage_prevention:
       drop_cols:
         - "Career_Merit_Cap_Share"
     ```
   * **`conf/split/chronological.yaml`**: Isolate partitioning logic.
     ```yaml
     strategy: "chronological"
     test_split_pct: 0.20
     right_censor_buffer: 20
     ```
   * **`conf/data/fof8_base.yaml`**: Isolate physical paths. Remove ML target definitions.
     ```yaml
     league_name: "DRAFT005"
     raw_path: "../fof8-gen/data/raw"
     features_path: "./data/processed/X_features.parquet"
     targets_path: "./data/processed/y_targets.parquet"
     active_team_id: null
     ```

2. **Refactor Pipeline Orchestrator:**
   * **`conf/economic_pipeline.yaml`**: Update the `defaults` list to compose the modules above. Remove all inline settings that have been migrated.
     ```yaml
     defaults:
       - data: fof8_base
       - target: economic
       - split: chronological
       - cv: kfold_5 # Assuming this exists or create a default
       - model@stage1_model: s1_catboost
       - model@stage2_model: reg_tweedie
       - _self_

     experiment_name: "Economic_Talent_Engine_v3"
     train_stage2: true
     use_gpu: false
     seed: 42
     ```

---

## Phase 3: Pipeline Decoupling (Core vs. Experimentation)
**Goal:** Create a strict boundary where `fof8-core` generates raw data, a transform script materializes it, and `train_pipeline.py` only consumes it.

1. **The Materializer (`src/fof8_ml/data/transform.py`):**
   * Update this script to utilize `build_economic_dataset` from `dataset.py`.
   * **Crucial Change:** Save the output as *two distinct artifacts*: `X_features.parquet` and `y_targets.parquet`. Do not join them on disk.
   * Ensure these artifacts are tracked via DVC.

2. **The Experimenter (`src/fof8_ml/train_pipeline.py`):**
   * Refactor the `main()` function to read both `X_features.parquet` and `y_targets.parquet` using the paths defined in `cfg.data`.
   * Perform the `[Player_ID, Year]` join dynamically in memory using `polars`.
   * Apply the runtime `split` configuration (`cfg.split`).
   * Define the `target_cols` to strip out dynamically based on the config: `[cfg.target.stage1_sieve.target_col, cfg.target.stage2_intensity.target_col] + list(cfg.target.leakage_prevention.drop_cols)`.
   * **Refactoring Note (Data Prep):** Extract the `polars` data prep and joining logic into a new helper function `prepare_economic_datasets()`. Place this function in `src/fof8_ml/data/dataset.py`.
   * **Refactoring Note (Training Loops):** Decouple the Stage 1 and Stage 2 training loops into standalone functions (`train_sieve_classifier()`, `train_intensity_regressor()`).
   * **CRITICAL:** You must preserve all existing `mlflow` logging (metrics, params, model artifacts) and Optuna dashboard tagging exactly as it functions currently. Do not remove tracking logic during the extraction.
