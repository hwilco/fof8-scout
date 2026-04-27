# FOF8 Experimentation

This package contains the formal machine learning training pipeline and experimentation notebooks for the FOF8 Draft Analyzer.

## Modular Architecture

The `fof8_ml` source code is organized into modular components to support scalability and easy experimentation with different model architectures:

- **`data/`**: Dataset construction, caching, and ML-specific feature transformations.
- **`models/`**: Wrapper classes that provide a unified API for different ML libraries (XGBoost, CatBoost, Scikit-Learn).
- **`evaluation/`**: Metrics calculation and standardized plotting (Feature Importance, Confusion Matrices, SHAP).
- **`train_pipeline.py`**: The main orchestrator for multi-stage hurdle modeling and cross-validation.

## Machine Learning Pipeline

The pipeline utilizes **DVC** for data versioning, **Hydra** for configuration management, and **MLflow** for experiment tracking.

### 1. Fetching the Data (DVC)

Raw simulation data is versioned via DVC and stored in `fof8-gen/data/raw` (referenced by this module). Ensure you have the latest data snapshot before running experiments:

```bash
# Pull the latest data tracked by the current Git commit
uv run dvc pull
```

> [!TIP]
> This command will pull data into `../fof8-gen/data/raw`. The ML pipeline is already configured to look for data at that path.

### 2. Running Training Experiments (Hydra)
Execute all ML commands from the monorepo root to ensure proper workspace resolution. We use `train_pipeline.py` for the primary Economic Talent Engine:

```bash
# Run the full 2-stage pipeline with default settings
uv run --package fof8-experimentation python src/fof8_ml/train_pipeline.py

# Override hyperparameters on the fly (e.g., Stage 1 learning rate)
uv run --package fof8-experimentation python src/fof8_ml/train_pipeline.py stage1_model.params.learning_rate=0.01
```

### Feature Ablation
You can experiment with different feature subsets using the `include_features` and `exclude_features` configuration options. Both support wildcards (e.g., `Delta_*`). If both are provided, `exclude_features` acts as a final filter (removing features even if they were explicitly included).

```bash
# Only use specific features
uv run --package fof8-experimentation python src/fof8_ml/train_pipeline.py data.include_features="[Combine_40, Combine_Bench]"

# Use all features EXCEPT specific ones
uv run --package fof8-experimentation python src/fof8_ml/train_pipeline.py data.exclude_features="[Weight, Age]"

# Wildcard support
uv run --package fof8-experimentation python src/fof8_ml/train_pipeline.py data.exclude_features="[Delta_*]"
```

### 3. Viewing Results (MLflow)
Every run automatically logs hyperparameters, metrics, and models to a centralized database and artifact store within this package. To view the dashboard:

```bash
# Start the local tracking server
uv run mlflow ui --backend-store-uri sqlite:///fof8-experimentation/mlflow.db --port 5000
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

