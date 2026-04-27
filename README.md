# fof8-scout

A `uv` workspace monorepo for scouting FOF8 draft prospects — automating data generation, exploring the data, and training ML models to predict draft prospect career value.

## Modules

| Module | Purpose | Runs On |
|---|---|---|
| [`fof8-gen/`](./fof8-gen/README.md) | RPA automation to simulate FOF8 seasons and export CSVs | **Windows host** |
| [`fof8-core/`](./fof8-core/README.md) | Shared domain logic, Polars pipelines, and schemas | **Hybrid** |
| [`fof8-experimentation/`](./fof8-experimentation/) | Polars + Jupyter data exploration and analysis | **Dev Container** |

> [!IMPORTANT]
> `fof8-gen` requires direct Windows desktop access (screen capture, GUI automation) and **cannot** run in a container. `fof8-experimentation` is fully containerized for reproducibility. `fof8-core` is a shared library.

---

## Repository Structure

```
fof8-scout/
├── fof8-gen/                 # RPA automation & data collection
│   ├── pyproject.toml
│   └── src/
├── fof8-core/                # Shared logic, schemas, and pipelines
│   ├── pyproject.toml
│   └── src/
├── fof8-experimentation/     # Data science & ML modeling
│   ├── conf/                # Hydra hierarchical configurations
│   ├── mlruns/              # MLflow centralized artifact store
│   ├── mlflow.db            # MLflow experiment metadata
│   ├── outputs/             # Organized run logs & local files
│   ├── multirun/            # Results from hyperparameter sweeps
│   ├── notebooks/           # Analysis and exploration notebooks
│   ├── src/fof8_ml/         # Modular ML pipeline
│   │   ├── data/            # Dataset & Transform logic
│   │   ├── models/          # Multi-library Model Wrappers
│   │   ├── evaluation/      # Metrics & Plotting
│   │   └── train_pipeline.py # Main orchestrator
│   └── pyproject.toml

├── .devcontainer/
│   └── experimentation/      # Dev Container config for experimentation only
├── pyproject.toml            # uv workspace root
└── uv.lock                   # Shared lock file for all members
```

---

## Getting Started

### fof8-gen (Windows host)

See the [fof8-gen README](./fof8-gen/README.md) for full setup and usage.

**Quick setup** (from the repo root):
```powershell
uv sync --package fof8-gen
```

### fof8-experimentation (Dev Container)

Open the repo in VS Code and select **"Reopen in Container"** when prompted (or run `Dev Containers: Reopen in Container` from the command palette). The container will install dependencies automatically via `postCreateCommand`.

#### GPU Acceleration (Optional)
To use GPU acceleration for model training (XGBoost/CatBoost):
1.  **Host Requirement**: Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your Linux or WSL2 host.
2.  **Configuration**: The `.devcontainer` is already configured to request GPU access via `--gpus all`.
3.  **Verification**: Run `uv run --package fof8-experimentation python fof8-experimentation/scratch/check_gpu.py` inside the container.

## Data Synchronization (DVC)

This project uses **[DVC (Data Version Control)](https://dvc.org/)** to manage large simulation datasets. Since data is collected on a Windows host but analyzed in a Linux container, DVC acts as the bridge.

### The Workflow

This project uses a hybrid DagsHub/DVC architecture to ensure data lineage across environments.

1.  **Collection (Windows)**: `fof8-gen` exports raw CSVs to `fof8-gen/data/raw/`.
2.  **Versioning (Windows)**: Run `dvc add fof8-gen/data/raw` to update the data version, then `git commit` and `dvc push`.
3.  **Consumption (Dev Container)**: Run `dvc pull` to sync the latest raw data.
4.  **Orchestration (Dev Container)**: Run `dvc repro fof8-experimentation/dvc.yaml` from the root. This will:
    -   **Transform**: Run `transform.py` to build a single "Universal Truth" feature store (`features.parquet`).
    -   **Train**: Run `train_pipeline.py` which dynamically splits the parquet in-memory and logs results to MLflow.

> [!TIP]
> The `git_commit` is automatically logged as a tag in MLflow, allowing you to trace any model run back to its exact data and code version.


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
