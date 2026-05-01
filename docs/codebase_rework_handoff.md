# FOF8 Scout Codebase Rework Handoff

## Purpose

This document converts the codebase review into an implementation plan that can be handed to one agent or split across several agents. The goal is to improve organization, readability, and extensibility without changing model behavior unless a change is explicitly called out and covered by tests.

The current architecture is mostly sound:

- `fof8-core` owns domain loading, schemas, features, and targets.
- `fof8-ml` owns dataset preparation, model wrappers, evaluation, and orchestration.
- `fof8-gen` owns Windows GUI automation and snapshot generation.
- `pipelines/` owns DVC/Hydra entrypoints.

The main work is to make boundaries explicit, split overloaded modules, reduce duplicated pipeline lifecycle code, and strengthen tests around extension points.

## Current Baseline

Known verification status from review:

- `uv run ruff check fof8-core fof8-ml fof8-gen pipelines scripts` passes.
- `uv run pytest fof8-core/tests fof8-ml/tests -q` has one failure:
  - `fof8-core/tests/test_data_pipelines.py::test_get_career_outcomes_pipeline`
  - The test calls `get_career_outcomes(..., final_year=2144)`, but `get_career_outcomes` now accepts only `loader`.
  - The test also expects old output columns such as `Was_Drafted`.

Do not start large refactors until the baseline test mismatch is corrected.

## Working Rules

- Keep behavior-preserving refactors separate from behavior changes.
- Prefer small PRs/commits by phase.
- Preserve DVC/Hydra CLI compatibility unless a migration note is included.
- Do not move raw data, generated data, notebooks, MLflow artifacts, or DVC-tracked outputs unless explicitly scoped.
- After every phase, run:
  - `uv run ruff check fof8-core fof8-ml fof8-gen pipelines scripts`
  - `uv run pytest fof8-core/tests fof8-ml/tests -q`
- For package moves, update imports, tests, README snippets, and DVC paths together.

## Recommended Parallelization

This work can be split among agents after Phase 0.

- Agent A: `fof8-core` feature/target/loader refactors.
- Agent B: `fof8-ml` dataset, training lifecycle, model registry refactors.
- Agent C: `fof8-gen` automation readability refactor.
- Agent D: tests/docs/verification and DVC/Hydra entrypoint compatibility.

Avoid parallel edits to the same files. In particular, only one agent should touch each of these at a time:

- `fof8-core/src/fof8_core/features.py`
- `fof8-core/src/fof8_core/targets.py`
- `fof8-ml/src/fof8_ml/data/dataset.py`
- `fof8-ml/src/fof8_ml/orchestration/data_loader.py`
- `pipelines/train_classifier.py`
- `pipelines/train_regressor.py`

## Phase 0: Restore Baseline Tests

### Scope

Fix stale tests so the suite reflects current intended behavior.

### Files

- `fof8-core/tests/test_data_pipelines.py`
- Potentially `fof8-core/src/fof8_core/targets.py` only if the old behavior is still desired.

### Tasks

1. Update `test_get_career_outcomes_pipeline` to call `get_career_outcomes(mock_loader)` with no `final_year`.
2. Update expected columns and shape to match current behavior:
   - Current output includes `Player_ID`, `Career_Games_Played`, `Championship_Rings`, `Hall_of_Fame_Flag`, `Number_of_Seasons`.
   - Current output does not include `Was_Drafted`.
3. Add a regression check that scanning all years keeps the latest row per `Player_ID`.
4. Confirm null career values are filled with zero.

### Acceptance Criteria

- Full test suite passes.
- No production behavior changes unless explicitly justified.

## Phase 1: Split `fof8-core` Feature Code

### Problem

`fof8-core/src/fof8_core/features.py` currently mixes:

- Draft-class loading.
- Combine z-score engineering.
- scouting mean/delta engineering.
- Staff scouting features.
- Position-specific feature masking constants and logic.

This makes the module harder to scan and creates a poor extension point for new feature groups.

### Proposed Structure

Create:

```text
fof8-core/src/fof8_core/
  features/
    __init__.py
    draft_class.py
    position_masks.py
    constants.py
```

If converting `features.py` to a package is too disruptive, use this lower-risk structure first:

```text
fof8-core/src/fof8_core/
  features.py
  feature_constants.py
  position_masks.py
```

Recommended approach: use the lower-risk structure first to avoid import ambiguity between `features.py` and a `features/` package.

### Tasks

1. Move feature group constants and `POSITION_FEATURE_MAP` to `position_masks.py` or `feature_constants.py`.
2. Move `apply_position_mask` to `position_masks.py`.
3. Keep `get_draft_class` import-compatible from `fof8_core.features`.
4. Re-export `apply_position_mask` from `fof8_core.features` or update callers:
   - `fof8-ml/src/fof8_ml/orchestration/data_loader.py`
5. Extract helper functions inside `get_draft_class`:
   - `_add_age_features`
   - `_add_position_relative_z_scores`
   - `_add_scouting_uncertainty_features`
   - `_add_staff_scouting_features`

### Acceptance Criteria

- Existing public imports still work or are migrated consistently.
- `get_draft_class` behavior is unchanged.
- Position masking behavior is unchanged.
- Tests pass.

## Phase 2: Add Target Registry and Split Target Logic

### Problem

`fof8-core/src/fof8_core/targets.py` already has a TODO for a target registry. Target generation is becoming a natural plugin surface.

### Proposed Structure

```text
fof8-core/src/fof8_core/
  targets.py                  # compatibility exports
  target_registry.py
  targets_financial.py
  targets_career.py
  targets_awards.py
  targets_av.py
```

### Tasks

1. Add a minimal registry:

```python
TARGET_REGISTRY: dict[str, Callable[[FOF8Loader], pl.DataFrame]] = {}

def register_target(name: str):
    ...

def get_target(name: str, loader: FOF8Loader, **kwargs) -> pl.DataFrame:
    ...
```

2. Register existing target builders:
   - `career_outcomes`
   - `annual_financials`
   - `peak_overall`
   - `merit_cap_share`
   - `career_value_metrics`
   - `awards` if static parameters are not required, or support partial configs.
3. Keep existing function names importable from `fof8_core.targets`.
4. Add tests for:
   - successful registry lookup
   - unknown target name error
   - registered function returns expected frame for a mocked loader

### Acceptance Criteria

- Existing callers continue to work.
- New config-driven target resolution is possible.
- Target modules are easier to navigate by domain.

## Phase 3: Clarify Dataset Construction vs Training Slicing

### Problem

`fof8-ml/src/fof8_ml/data/dataset.py` mixes durable feature-store construction with training-time concerns. It also contains legacy survival dataset code with an unresolved TODO.

### Proposed Structure

```text
fof8-ml/src/fof8_ml/data/
  dataset.py                  # compatibility exports
  economic_dataset.py
  survival_dataset.py         # deprecated or explicitly supported
  categorical.py
```

### Tasks

1. Extract shared categorical preparation:
   - rare college bucketing
   - string/categorical to `pl.Enum`
2. Decide whether `build_survival_dataset` is supported or deprecated:
   - If supported, move to `survival_dataset.py` and add tests.
   - If deprecated, keep a wrapper in `dataset.py` with a deprecation warning and update scripts such as `scripts/plot_probs.py`.
3. Keep `build_economic_dataset` focused on building the universal feature/target/metadata frame.
4. Keep chronological splitting in `fof8_ml.orchestration.data_loader`, not in dataset builders.
5. Add explicit return type dataclasses if useful:
   - `DatasetBuildResult`
   - `FeatureTargetFrames`

### Acceptance Criteria

- `pipelines/process_features.py` still builds `features.parquet`.
- Training scripts continue to consume the same parquet schema.
- Tests cover categorical bucketing and enum conversion.

## Phase 4: Make DataLoader Cache Deterministic

### Problem

`DataLoader.load` uses `hash(str(data_cfg))`, and Python hash randomization makes this unstable across processes.

### Tasks

1. Replace with deterministic hashing:

```python
import hashlib
import json

cfg_hash = hashlib.sha256(
    json.dumps(data_cfg, sort_keys=True, default=str).encode("utf-8")
).hexdigest()
```

2. Consider moving the cache into an instance-owned object:
   - Current module global cache is acceptable for Hydra sweeps but harder to test.
   - A `DataCache` class would make behavior explicit.
3. Add tests for:
   - same config produces same hash
   - changed threshold/positions/split produces different hash
   - feature ablation does not poison the base cache

### Acceptance Criteria

- Sweep cache behavior remains intact.
- Cache key is reproducible.
- Tests document what config fields affect cache reuse.

## Phase 5: Consolidate Classifier and Regressor Pipeline Lifecycle

### Problem

`pipelines/train_classifier.py` and `pipelines/train_regressor.py` duplicate setup and teardown:

- experiment root resolution
- `ExperimentLogger`
- `SweepManager`
- `DataLoader`
- feature ablation
- MLflow parent run
- optimization metric selection
- champion update
- DVC metrics write

### Proposed Structure

```text
fof8-ml/src/fof8_ml/orchestration/
  pipeline_runner.py
  stage_classifier.py
  stage_regressor.py
```

Keep root scripts as thin Hydra entrypoints initially:

```text
pipelines/train_classifier.py
pipelines/train_regressor.py
```

### Tasks

1. Introduce a `PipelineContext` dataclass:
   - `cfg`
   - `exp_root`
   - `absolute_raw_path`
   - `logger`
   - `sweep_mgr`
   - `sweep_context`
   - `data`
2. Introduce shared setup:
   - `build_pipeline_context(cfg)`.
3. Introduce shared completion:
   - `select_optimization_metric(available_metrics, cfg.optimization.metric)`
   - `finalize_pipeline_run(...)`
4. Move classifier-specific body to `run_classifier_stage(ctx)`.
5. Move regressor-specific body to `run_regressor_stage(ctx)`.
6. Keep existing CLI/Hydra config names stable.

### Acceptance Criteria

- `uv run python pipelines/train_classifier.py --help` still works.
- `uv run python pipelines/train_regressor.py --help` still works.
- DVC commands in `dvc.yaml` still work.
- Duplicate lifecycle code is reduced meaningfully.

## Phase 6: Add Model Registry

### Problem

`fof8_ml.models.factory.get_model_wrapper` uses substring matching. This is workable now but will become brittle as model options grow.

### Proposed Structure

```text
fof8-ml/src/fof8_ml/models/
  registry.py
  factory.py                  # compatibility wrapper around registry
```

### Tasks

1. Define a registry keyed by `(stage, model_key)` or model config name.
2. Register current wrappers:
   - stage1 catboost
   - stage1 xgb
   - stage2 catboost
   - stage2 xgb
   - stage2 sklearn/tweedie/gamma
3. Replace substring matching with explicit aliases:
   - `s1_catboost`
   - `survival_xgb`
   - `reg_tweedie`
   - `reg_gamma`
4. Keep friendly errors listing valid model names.
5. Add tests for each configured model YAML resolving to a wrapper.

### Acceptance Criteria

- Existing model configs resolve.
- Adding a new model requires one registration entry and one config file.
- Tests fail clearly if a config references an unknown model.

## Phase 7: Train/Inference Schema Contract

### Problem

`pipelines/batch_inference.py` manually recreates preprocessing and category handling. This risks training/inference drift.

### Tasks

1. During training, persist a feature schema artifact:
   - feature column names
   - categorical columns
   - enum/category handling policy
   - dropped metadata/target columns
   - college bucketing policy, if applicable
2. Add a `FeaturePreprocessor` or `FeatureSchema` object in `fof8_ml.data`.
3. Use the same object in:
   - training data prep
   - batch inference
4. Update `batch_inference.py` to load schema/preprocessor from MLflow artifacts when loading the model.
5. Add tests for missing/extra inference columns:
   - columns are ordered consistently
   - missing optional columns get safe defaults only if policy allows
   - unexpected columns are dropped or rejected according to policy

### Acceptance Criteria

- Batch inference no longer hardcodes category handling independently.
- Training and inference use the same feature ordering.
- Schema mismatch produces actionable errors.

## Phase 8: Refactor `fof8-gen` Automation Flow

### Problem

`fof8-gen/src/fof8_gen/automation.py` is a long procedural script. It is hard to change individual GUI steps safely.

### Proposed Structure

```text
fof8-gen/src/fof8_gen/
  automation.py               # CLI compatibility
  automation_runner.py
  screen.py
  workflows.py
```

### Tasks

1. Move image waiting/clicking to `screen.py`.
2. Wrap `prevent_sleep`/`allow_sleep` in a context manager:

```python
@contextmanager
def prevent_system_sleep():
    ...
```

3. Move metadata loading to its own function/module.
4. Convert the season loop into named workflow methods:
   - `export_draft_snapshot`
   - `create_one_year_history`
   - `export_post_sim_snapshot`
   - `advance_to_staff_draft`
   - `complete_staff_draft`
   - `begin_free_agency`
5. Keep `gather-data = "fof8_gen.automation:main"` working.

### Acceptance Criteria

- CLI behavior remains compatible.
- Each workflow step is individually testable with mocked `pyautogui`.
- Snapshot-only mode still works.

## Phase 9: Loader Cleanup

### Problem

`FOF8Loader` is mostly good, but `get_active_team_id` mixes metadata parsing, hardcoded lookup, and fallback CSV scanning.

### Tasks

1. Extract metadata parsing:
   - `read_metadata_team_id`
   - `read_metadata_team_name`
2. Move team name lookup to a constant:
   - `TEAM_NAME_TO_ID`
3. Prefer PyYAML if `fof8-core` can accept the dependency.
   - If not, use a small explicit parser and document the supported metadata fields.
4. Replace deprecated Polars `dtypes=` usage in `scan_csv` with `schema_overrides=`.
5. Cache discovered year directories if profiling shows repeated filesystem scans matter.

### Acceptance Criteria

- Loader tests still pass.
- Deprecation warning for `dtypes` disappears.
- Team ID behavior remains unchanged.

## Phase 10: Documentation and Tooling Updates

### Tasks

1. Update root `README.md` if paths move.
2. Update `fof8-core/README.md` examples if feature/target imports change.
3. Update `fof8-ml/README.md` to describe:
   - shared runner
   - model registry
   - train/inference schema artifact
4. Add or update architecture notes:
   - package responsibilities
   - extension guide for adding a target
   - extension guide for adding a model
5. Consider adding Pyright config and running it in CI/pre-commit after refactors stabilize.

### Acceptance Criteria

- New contributors can identify where to add a target, feature, model, or pipeline stage.
- README commands still work.

## Suggested PR/Commit Plan

1. `tests: align career outcome tests with current API`
2. `refactor(core): split position mask constants from draft features`
3. `refactor(core): add target registry with compatibility exports`
4. `refactor(ml): split economic and survival dataset builders`
5. `fix(ml): use deterministic data cache keys`
6. `refactor(ml): introduce shared training pipeline context`
7. `refactor(ml): add explicit model registry`
8. `feat(ml): persist feature schema for inference`
9. `refactor(gen): split GUI automation workflow into named steps`
10. `docs: update architecture and extension guides`

## Risk Register

### Import Compatibility

Moving modules can break notebooks and scripts. Keep compatibility exports during the transition and add deprecation warnings later.

### Hydra/DVC Path Stability

DVC references root-level pipeline scripts. Keep wrappers in `pipelines/` until DVC is deliberately migrated.

### Training Behavior Drift

Refactoring dataset prep and feature schema code can silently change model inputs. Add tests that compare pre/post refactor feature column lists and representative transformed frames.

### MLflow Artifact Compatibility

Changing model logging or artifact paths can break old inference scripts. Preserve current artifact names where possible, and version new schema artifacts.

### Windows Automation

`fof8-gen` requires a Windows GUI and cannot be fully integration-tested in the dev container. Use unit tests with mocked `pyautogui` and keep manual test instructions.

## Final Verification Checklist

Before considering the rework complete:

- `uv run ruff check fof8-core fof8-ml fof8-gen pipelines scripts`
- `uv run ruff format --check fof8-core fof8-ml fof8-gen pipelines scripts`
- `uv run pytest fof8-core/tests fof8-ml/tests -q`
- `uv run python pipelines/process_features.py --help` or equivalent Hydra help
- `uv run python pipelines/train_classifier.py --help`
- `uv run python pipelines/train_regressor.py --help`
- `uv run dvc repro --dry` if available
- Manual smoke test for `fof8-gen` on Windows after automation refactor
