# FOF8 Experimentation

This package contains the formal machine learning training pipeline and experimentation notebooks for the FOF8 Draft Analyzer.

## Modular Architecture

The `fof8_ml` source code is organized into modular components to support scalability and easy experimentation with different model architectures:

- **`data/`**: Dataset construction, caching, and ML-specific feature transformations.
- **`models/`**: Wrapper classes that provide a unified API for different ML libraries (XGBoost, CatBoost, Scikit-Learn) and a **Model Factory**.
- **`evaluation/`**: Metrics calculation and standardized plotting (Feature Importance, Confusion Matrices, SHAP).
- **`orchestration/`**: The core logic for pipeline execution (DataLoader, Trainer, Logger, and Sweep Management).

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

## Output Organization

To keep the source tree clean, all execution artifacts are organized as follows:

- **`mlflow.db`**: Centralized SQLite database for experiment tracking.
- **`mlruns/`**: Centralized artifact store (models, plots, CSVs).
- **`outputs/`**: Timestamped logs and local temporary files for individual runs.
- **`multirun/`**: Organized results for Optuna hyperparameter sweeps.

> [!TIP]
> All local logs and generated files (like OOF results) are automatically saved into the timestamped `outputs/` directory for each run, so your project root stays clean!
