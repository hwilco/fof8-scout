# fof8-scout

A `uv` workspace monorepo for scouting FOF8 draft prospects — automating data generation, exploring the data, and training ML models to predict draft prospect career value.

## Modules

| Module | Purpose | Runs On |
|---|---|---|
| [`fof8-gen/`](./fof8-gen/README.md) | RPA automation to simulate FOF8 seasons and export CSVs | **Windows host** |
| [`fof8-core/`](./fof8-core/README.md) | Shared domain logic, Polars pipelines, and schemas | **Hybrid** |
| [`fof8-ml/`](./fof8-ml/README.md) | Modular ML modeling pipeline | **Dev Container** |
| [`notebooks/`](./notebooks/) | Jupyter data exploration and analysis | **Dev Container** |

> [!IMPORTANT]
> `fof8-gen` requires direct Windows desktop access (screen capture, GUI automation) and **cannot** run in a container. `fof8-ml` is fully containerized for reproducibility. `fof8-core` is a shared library.

---

## Repository Structure

```
fof8-scout/
├── fof8-gen/                 # RPA automation & data collection (Windows)
│   ├── data/                 # Raw exported CSVs (DVC tracked)
│   ├── pyproject.toml
│   └── src/                  # GUI automation logic
├── fof8-core/                # Shared logic, schemas, and pipelines (Hybrid)
│   ├── pyproject.toml
│   └── src/fof8_core/        # Shared Polars logic & feature engineering
├── fof8-ml/                  # Data science & ML modeling (Dev Container)
│   ├── mlruns/               # MLflow centralized artifact store
│   ├── mlflow.db             # MLflow experiment metadata
│   ├── optuna.db             # Optuna persistent study storage
│   ├── src/fof8_ml/          # Modular ML pipeline components
│   │   ├── data/             # Dataset builders + train/inference feature schema
│   │   ├── models/           # Stage-aware model wrappers + explicit registry
│   │   ├── evaluation/       # Metrics & plotting
│   │   └── orchestration/    # Shared pipeline context + stage runners
├── pipelines/                # ML Orchestration Scripts (DVC Stages)
│   ├── conf/                 # Hydra Hierarchical Configs
│   ├── process_features.py   # Feature store builder
│   ├── train_classifier.py   # Sieve Classifier (S1) orchestration
│   ├── train_regressor.py    # Intensity Regressor (S2) orchestration
│   └── batch_inference.py    # Prediction pipeline
├── notebooks/                # Analysis and exploration notebooks
├── docs/                     # Detailed technical documentation
├── scripts/                  # Utility scripts and validation tools
├── outputs/                  # Hydra local run logs (root-level by default)
├── multirun/                 # Hydra multirun/sweep logs
├── .devcontainer/            # Reproducible experimentation environment
├── dvc.yaml                  # DVC pipeline definition
├── pyproject.toml            # uv workspace root
└── uv.lock                   # Shared lock file for all members
```

---

## Getting Started

### Dependency Profiles

Use dependency groups explicitly based on the task:

```powershell
# Root development tooling (ruff/pyright/pytest/pre-commit/nbstripout)
uv sync --group dev

# Root MLOps tooling (mlflow, dagshub, dvc[s3])
uv sync --group mlops

# fof8-ml runtime dependencies only (no notebook/viz stack)
uv sync --package fof8-ml

# fof8-ml with notebook + visualization tooling
uv sync --package fof8-ml --group notebook --group viz
```

### fof8-gen (Windows host)

See the [fof8-gen README](./fof8-gen/README.md) for full setup and usage.

**Quick setup** (from the repo root):
```powershell
uv sync --package fof8-gen
```

### fof8-ml (Dev Container)

Open the repo in VS Code and select **"Reopen in Container"** when prompted (or run `Dev Containers: Reopen in Container` from the command palette). The container will install dependencies automatically via `postCreateCommand`.

#### GPU Acceleration (Optional)
To use GPU acceleration for model training (XGBoost/CatBoost):
1.  **Host Requirement**: Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your Linux or WSL2 host.
2.  **Configuration**: The `.devcontainer` is already configured to request GPU access via `--gpus all`.
3.  **Verification**: Run `uv run --package fof8-ml python fof8-ml/scratch/check_gpu.py` inside the container.

## Data Synchronization (DVC)

This project uses **[DVC (Data Version Control)](https://dvc.org/)** to manage large simulation datasets. Since data is collected on a Windows host but analyzed in a Linux container, DVC acts as the bridge.

### The Workflow

This project uses a hybrid DagsHub/DVC architecture to ensure data lineage across environments.

1.  **Collection (Windows)**: `fof8-gen` exports raw CSVs to `fof8-gen/data/raw/`.
2.  **Versioning (Windows)**: Run `dvc add fof8-gen/data/raw` to update the data version, then `git commit` and `dvc push`.
3.  **Consumption (Dev Container)**: Run `dvc pull` to sync the latest raw data.
4.  **Orchestration (Dev Container)**: Run `dvc repro` from the root. This will:
    -   **Transform**: Run `pipelines/process_features.py` to build a single "Universal Truth" feature store (`features.parquet`).
    -   **Train**: Run `pipelines/train_classifier.py` and/or `pipelines/train_regressor.py`. These stages dynamically split the parquet in-memory and log independent results to MLflow.

> [!TIP]
> The `git_commit` is automatically logged as a tag in MLflow, allowing you to trace any model run back to its exact data and code version.

## Extending The System

See [`docs/architecture_extensions.md`](./docs/architecture_extensions.md) for:
- package responsibilities
- adding a new target (`fof8_core.targets.registry`)
- adding a new model (`fof8_ml.models.registry`)
- adding a new feature group (`fof8_core.features`)

---

## Hyperparameter Tuning & Warm Starting

This project uses **Optuna** for hyperparameter sweeps. By default, sweeps start with no prior knowledge. To enable "Warm Starting" (learning from past results) and "Resuming" interrupted sweeps:

1.  **Persistent Storage**: Sweeps use a local SQLite database at `fof8-ml/optuna.db` to store the optimizer's state.
2.  **How to Warm Start**:
    - Ensure your experiment config (e.g., `s1_catboost_sweep.yaml`) has `storage: sqlite:///fof8-ml/optuna.db`.
    - Use a consistent `study_name`.
    - When you rerun the sweep, Optuna will automatically load previous trials from the DB and use them to inform the next sampling decision.
3.  **Resetting**: To start a sweep completely fresh, delete the `fof8-ml/optuna.db` file.

---

## Development Tooling

All modules share root-level dev tooling configured in the root `pyproject.toml`:

- **[Ruff](https://docs.astral.sh/ruff/)** — linting and formatting
- **[Pyright](https://github.com/microsoft/pyright)** — static type checking

```powershell
# Lint and auto-fix
uv run ruff check --fix

# Format
uv run ruff format
```

### Git Hygiene & Pre-commit

This project uses `nbstripout` to automatically strip outputs from Jupyter Notebooks in Git while keeping them intact on your local machine, and `pre-commit` to ensure code quality.

#### 1. Notebook Stripping (`nbstripout`)

The Dev Container handles this automatically. If you are working on the **Windows host**, run these commands once to set up a portable Git filter that works across both environments:

```powershell
uv run nbstripout --install
git config filter.nbstripout.clean "uv run nbstripout"
git config diff.ipynb.textconv "uv run nbstripout -t"
```

#### 2. Pre-commit Hooks

To install on your host, run:

```powershell
uv run pre-commit install
```

Hooks are installed automatically in the Dev Container. They run automatically every time you `git commit`.

To run all checks manually against the entire repository:

```powershell
uv run pre-commit run --all-files
```
