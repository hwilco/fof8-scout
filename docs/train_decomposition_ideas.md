# Train Pipeline Decomposition Ideas

The `pipelines/train.py` script is currently a large monolith orchestrator (~800 lines). While it successfully integrates Hydra, DVC, and MLflow, its size makes it difficult to maintain and test.

## Current Challenges
- **Monolithic Execution**: Data loading, cross-validation, feature ablation, metric calculation, threshold optimization, and logging are all tangled in a single `main()` function.
- **Optuna Sweep Overhead**: The hyperparameter sweep logic clutters the core training loop.
- **Difficult to Unit Test**: Because everything is tightly coupled to MLflow runs and Hydra configs, unit testing individual steps (like threshold tuning or fold splitting) is nearly impossible.

## Proposed Decomposition Strategy

We should break `pipelines/train.py` into focused, testable modules under a new `fof8_ml/orchestration/` or `pipelines/components/` directory.

### 1. Data Orchestrator (`fof8_ml/orchestration/data_loader.py`)
Extract the logic that parses Hydra configs (`league_name`, `year_range`, etc.), calls `fof8_core.loader`, applies position masks, filters targets, and interfaces with the caching module.
**Output**: A clean `(X_train, y_train, meta_train), (X_test, y_test, meta_test)` tuple.

### 2. Trainer Engine (`fof8_ml/orchestration/trainer.py`)
A class or function responsible for executing the cross-validation loop.
- Takes the instantiated model wrappers, data, and CV config.
- Handles the `StratifiedKFold` splits.
- Calls `model.fit()` and generates OOF predictions.
**Output**: OOF predictions, trained models, and out-of-fold indices.

### 3. Evaluator (`fof8_ml/orchestration/evaluator.py`)
Decouple threshold optimization and metric calculation from the training loop.
- Takes OOF predictions and ground truth.
- Optimizes classification thresholds (e.g., maximizing F1 score).
- Computes `calculate_survival_metrics`.
**Output**: A clean dictionary of metrics and the optimal threshold.

### 4. Logger (`fof8_ml/orchestration/logger.py`)
Centralize MLflow API calls.
- Wrap `mlflow.log_metrics`, `mlflow.log_params`, and artifact logging.
- Handle DagsHub model registry registration.

### Resulting `train.py`
The final `pipelines/train.py` should just be a clean DAG of these components:
```python
def main(cfg):
    X, y, meta = data_loader.load(cfg)
    models = trainer.train_cv(cfg.models, X, y)
    metrics, threshold = evaluator.evaluate(models.oof_preds, y)
    logger.log_experiment(cfg, models, metrics, threshold)
```
