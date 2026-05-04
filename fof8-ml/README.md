# FOF8 Experimentation

This package contains the formal machine learning training pipeline and experimentation notebooks for the FOF8 Draft Analyzer.

## Dependency Setup

Install only runtime requirements for training/inference:

```bash
uv sync --package fof8-ml
```

Install runtime plus notebook/visualization tooling:

```bash
uv sync --package fof8-ml --group notebook --group viz
```

## Modular Architecture

The `fof8_ml` source code is organized into modular components to support scalability and easy experimentation with different model architectures:

- **`data/`**: Dataset construction, caching, and ML-specific feature transformations.
- **`models/`**: Wrapper classes and an explicit role-aware **Model Registry** for deterministic model resolution.
- **`evaluation/`**: Metrics calculation and standardized plotting (Feature Importance, Confusion Matrices, SHAP).
- **`orchestration/`**: Shared pipeline lifecycle (`PipelineContext`, loader/logger/sweeps) and role runners.

## Machine Learning Pipeline

The pipeline utilizes **DVC** for data versioning, **Hydra** for configuration management, and **MLflow** for experiment tracking.

### 1. Data Pipeline Orchestration (DagsHub)

We use DVC to manage the "Universal Truth" data store and orchestrate the training pipeline. This ensures that features are only reprocessed when the raw data or transformation logic changes.

```bash
# From the project root:
# 1. Pull the latest raw data
uv run dvc pull

# 2. Run the full pipeline (Transform -> Train)
uv run dvc repro
```

### 2. Running Training Experiments (Hydra)

The training pipeline is decoupled from the data transformation. You can run either the classifier or regressor script directly for fast iteration:

```bash
# Run the Sieve Classifier with a specific override
uv run python pipelines/train_classifier.py experiment_name="Testing_Classifier"

# Run the Intensity Regressor
uv run python pipelines/train_regressor.py experiment_name="Testing_Regressor"
```

> [!IMPORTANT]
> **Data vs ML Decoupling**:
> - The `transform` stage (handled by DVC) builds the single `features.parquet` file.
> - The `train` stage handles chronological splitting **in-memory**. This means you can adjust your test split cutoff in `conf/data/fof8_base.yaml` and rerun `dvc repro`—DVC will skip the expensive transformation and only rerun the training.

### Shared Pipeline Runner
`pipelines/train_classifier.py` and `pipelines/train_regressor.py` are thin Hydra entrypoints.
Common setup/teardown is centralized in `fof8_ml.orchestration.pipeline_runner`:
- `build_pipeline_context(...)`
- `select_optimization_metric(...)`
- `finalize_pipeline_run(...)`

### Adding A Model
Model resolution is explicit and registry-based (not substring-based).

To add a model:
- Add one model config in `pipelines/conf/model/*.yaml`.
- Add one registry entry in `fof8_ml.models.registry` for the correct role key.
- Use the registered config `name` in your Hydra override or experiment config.

If a model key is missing, training fails with a clear error listing valid keys for that role.

### Feature Ablation
You can experiment with different feature subsets using the `include_features` and `exclude_features` configuration options. Both support wildcards (e.g., `Delta_*`). If both are provided, `exclude_features` acts as a final filter (removing features even if they were explicitly included).

```bash
# Example: Only use specific features for the classifier
uv run python pipelines/train_classifier.py include_features="[Combine_40, Combine_Bench]"

# Example: Exclude specific features for the regressor
uv run python pipelines/train_regressor.py exclude_features="[Weight, Age]"

# Wildcard support
uv run python pipelines/train_classifier.py exclude_features="[Delta_*]"
```

### 3. Viewing Results (MLflow)
Every run automatically logs hyperparameters, metrics, and models to a centralized database and artifact store within this package. To view the dashboard:

```bash
# Start the local tracking server
uv run mlflow ui --backend-store-uri sqlite:///fof8-ml/mlflow.db --port 5000
```
Then open `http://localhost:5000` in your browser.

Classifier and regressor training both log a train/inference feature contract artifact at the MLflow run artifact root:

- `feature_schema.json`
- `classifier_model/` for classifier runs
- `regressor_model/` for regressor runs

`feature_schema.json` is not logged inside the model artifact directory. Load it from the same run ID as the model, for example `runs:/<classifier_run_id>/feature_schema.json` and `runs:/<classifier_run_id>/classifier_model`.

Batch inference loads this artifact and raises explicit schema mismatch errors for missing required columns or incompatible feature sets.

### Complete Classifier + Regressor Evaluation

The trained classifier and regressor are separate MLflow model artifacts with separate feature contracts. A complete expected-value model should load each model with its own `feature_schema.json`, prepare the full feature frame twice, and combine only the prediction arrays:

```text
complete_prediction(X) = P(clear_sieve | X) * E(positive_economic_value | clear_sieve, X)
```

```python
import json

import mlflow
import numpy as np
import polars as pl
from fof8_ml.data.schema import FEATURE_SCHEMA_ARTIFACT_PATH, FeatureSchema


def load_feature_schema(client: mlflow.tracking.MlflowClient, run_id: str) -> FeatureSchema:
    schema_path = client.download_artifacts(run_id, FEATURE_SCHEMA_ARTIFACT_PATH)
    with open(schema_path) as f:
        return FeatureSchema.from_dict(json.load(f))


def load_complete_model(
    classifier_run_id: str,
    regressor_run_id: str,
) -> tuple[object, FeatureSchema, object, FeatureSchema]:
    client = mlflow.tracking.MlflowClient()

    classifier_schema = load_feature_schema(client, classifier_run_id)
    regressor_schema = load_feature_schema(client, regressor_run_id)

    classifier = mlflow.catboost.load_model(
        f"runs:/{classifier_run_id}/classifier_model"
    )
    regressor = mlflow.catboost.load_model(
        f"runs:/{regressor_run_id}/regressor_model"
    )

    return classifier, classifier_schema, regressor, regressor_schema


def predict_complete_model(
    X_full: pl.DataFrame,
    classifier: object,
    classifier_schema: FeatureSchema,
    regressor: object,
    regressor_schema: FeatureSchema,
) -> np.ndarray:
    X_classifier = classifier_schema.apply(X_full)
    X_regressor = regressor_schema.apply(X_full)

    p_clear = classifier.predict_proba(X_classifier.to_pandas())[:, 1]
    value_if_clear = regressor.predict(X_regressor.to_pandas())

    return p_clear * np.maximum(value_if_clear, 0)
```

Key rules:

- Always use the classifier run's schema for the classifier and the regressor run's schema for the regressor.
- Do not force the two models to share a feature set unless they were trained that way.
- Preserve the schema's column order before prediction.
- Let the schema drop extra columns and reject missing required columns.
- For CatBoost Tweedie regressor runs, predictions are already in raw target space; do not apply `expm1`.
- If a future regressor is trained on log targets, convert its output back with `np.expm1` before stitching.

Default target-family setup:

- Classifier target: `Economic_Success`
- Regressor target: `Positive_Career_Merit_Cap_Share`
- Composite baseline target available for comparison: `DPO` via `fof8_core.targets.composite`

### Extension Guides
See [`docs/architecture_extensions.md`](../docs/architecture_extensions.md) for:
- adding a model (registry + config)
- adding a target
- adding a feature group

See [`docs/draft_value_target_strategy.md`](../docs/draft_value_target_strategy.md) for:
- choosing between overall-based and economic regression targets
- evaluating regressors with both draft-board ranking and value magnitude metrics
- the recommended target/metric setup for complete classifier + regressor model selection

See [`docs/draft_model_finish_line_plan.md`](../docs/draft_model_finish_line_plan.md) for:
- the concrete implementation plan to add target experiments, draft-aware metrics, complete-model evaluation, and final model-selection gates

## Output Organization

To keep the source tree clean, all execution artifacts are organized as follows:

- **`mlflow.db`**: Centralized SQLite database for experiment tracking.
- **`mlruns/`**: Centralized artifact store (models, plots, CSVs).
- **`outputs/`**: Timestamped logs and local temporary files for individual runs.
- **`multirun/`**: Organized results for Optuna hyperparameter sweeps.

> [!TIP]
> All local logs and generated files (like OOF results) are automatically saved into the timestamped `outputs/` directory for each run, so your project root stays clean!
