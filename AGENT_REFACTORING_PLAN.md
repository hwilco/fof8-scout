### 🤖 AI Agent Execution Instructions: Monorepo Reorganization

**Objective:** Refactor the `fof8-scout` monorepo into a modular data engineering and machine learning pipeline architecture.

**Constraints & Rules:**
1.  **Preserve History:** You MUST use `git mv` for all file and directory movements to preserve Git history.
2.  **No Deletions Without Confirmation:** Except for the `scratch/` directory cleanup specified below, do not permanently delete files without user approval.
3.  **Atomic Commits:** Pause and commit at the end of each Phase.
4.  **Tooling:** Assume the environment uses `uv` for dependency management and `dvc` for pipeline tracking.

---

#### **Phase 1: Foundation & Scaffolding**
1. Ensure the workspace is clean (`git status`).
2. Create a new branch: `git checkout -b refactor/modular-ml-pipeline`
3. Create the new root-level directories:
   ```bash
   mkdir -p pipelines pipelines/conf notebooks scripts
   ```

#### **Phase 2: Isolating the ML Library**
The `fof8-ml` directory currently acts as both a library and an execution space. We will re-scope it to be purely an ML library.
1. Rename the top-level directory:
   ```bash
   git mv fof8-ml fof8-ml
   ```
2. Update the workspace reference. In the root `pyproject.toml`, find the `members` array and change `"fof8-ml"` to `"fof8-ml"`.
3. In `fof8-ml/pyproject.toml`, verify the package name is correct (it should be `fof8_ml` or similar).

#### **Phase 3: Extracting Orchestration & Execution**
Move the specific pipeline executions out of the library and into the dedicated `pipelines/` directory.
1. Relocate configuration files:
   ```bash
   git mv fof8-ml/conf/* pipelines/conf/
   rm -rf fof8-ml/conf
   ```
2. **DO NOT move the DVC pipeline definitions.** Leave `dvc.yaml` and `dvc.lock` in the repository root. Moving them alters the relative pathing for the entire DAG.
3. Extract and rename the entry-point scripts:
   ```bash
   git mv fof8-ml/src/fof8_ml/train_pipeline.py pipelines/train.py
   git mv fof8-ml/src/fof8_ml/predict_draft.py pipelines/batch_inference.py
   ```

#### **Phase 4: Consolidating the Feature Store (`fof8-core`)**
Ensure that all domain-specific data transformation logic lives in one place to prevent training-serving skew.
1. Read the contents of `fof8-ml/src/fof8_ml/data/transforms.py`, `fof8-ml/src/fof8_ml/data/transform.py`, and `fof8-core/src/fof8_core/features.py`.
2. Migrate any feature calculation logic (e.g., positional versatility, cap-adjusted earnings) from the `fof8_ml` files into `fof8-core/src/fof8_core/features.py`.
3. Ensure `fof8_core.features` exposes standardized functions that strictly type-hint and utilize `polars.DataFrame` inputs and outputs. Do not use Pandas.
4. Once migrated, delete the redundant transformation scripts from `fof8-ml/src/fof8_ml/data/`.

#### **Phase 5: Quarantine Exploratory Code**
Separate temporary scripts and notebooks from the production path.
1. Move all Jupyter notebooks to the root `notebooks/` directory:
   ```bash
   git mv fof8-ml/notebooks/*.ipynb notebooks/
   rm -rf fof8-ml/notebooks
   ```
2. Audit `fof8-ml/scratch/` and `fof8-core/scratch/`.
   * Move reusable utility scripts (e.g., `check_gpu.py`, `plot_probs.py`) to the root `scripts/` directory.
   * Propose deletion for the remaining obsolete scripts in the `scratch/` folders. Do not attempt to convert any scripts into tests.

#### **Phase 6: Update Imports and Paths (CRITICAL)**
Because files have moved, import statements and relative paths are broken.
1. **Search and Replace:** Perform a search and replace for `fof8-ml` to `fof8-ml` strictly limited to `.md`, `.json`, `.yaml`, and `.py` files. Explicitly exclude the `.git/`, `.dvc/`, and `.venv/` directories from this operation to prevent repository corruption.
2. **Fix Entry Points:** In `pipelines/train.py` and `pipelines/batch_inference.py`, update local imports. They should now import models and evaluations from `fof8_ml` (which is in their Python path via `uv`), and feature logic from `fof8_core`.
3. **Fix Hydra Configs:** Ensure `pipelines/train.py` correctly points to `pipelines/conf/` for its Hydra configuration initialization.
   * *Agent instruction:* Modify the `@hydra.main(config_path="conf", ...)` decorator appropriately. Additionally, explicitly use `hydra.utils.get_original_cwd()` or ensure `hydra.job.chdir=True` is properly configured in the yaml files if any relative data paths are loaded within the training script, since DVC will execute from the root.
4. **Fix DVC:** Open the root `dvc.yaml` and update the `cmd` to execute `python pipelines/train.py` instead of the old path. Verify that the `deps` and `outs` paths remain correct relative to the root directory.

#### **Phase 7: Validation**
1. Run `uv sync` to ensure the workspace lockfile is updated with the new package names.
2. Run the test suite: `pytest fof8-core/tests fof8-ml/tests` to verify internal library logic remains intact.
3. Dry-run the pipeline: Ask the user to execute `dvc repro -s dvc.yaml` to ensure the DAG resolves correctly.
