# Economic Targets Architecture Cleanup Plan

## Goal

Clean up the economic target architecture after the initial refactor. Prefer a clear,
single-source target contract over backward compatibility shims. Older processed artifacts can
be regenerated instead of carrying duplicate target derivation logic indefinitely.

## Desired End State

- `fof8_core.targets` owns all economic target semantics, formulas, and target column lists.
- `fof8_ml` consumes target frames and target contracts, but does not redefine economic target
  formulas.
- Target profiles can pass parameters such as `merit_threshold` into target builders.
- Processed feature stores are expected to contain the current target schema. If they do not,
  users should re-run feature processing.
- Leakage prevention uses target-contract metadata instead of hand-maintained duplicate lists.

## Implementation Plan

### 1. Add Economic Target Constants

Create or extend a core module, preferably:

```text
fof8-core/src/fof8_core/targets/economic.py
```

Add explicit constants:

```python
ECONOMIC_TARGET_COLUMNS = [
    "Cleared_Sieve",
    "Economic_Success",
    "DPO",
    "Positive_DPO",
    "Career_Merit_Cap_Share",
    "Positive_Career_Merit_Cap_Share",
]

ECONOMIC_CONTEXT_COLUMNS = [
    "Peak_Overall",
    "Career_Games_Played",
]

ECONOMIC_OUTPUT_COLUMNS = [
    "Player_ID",
    *ECONOMIC_TARGET_COLUMNS,
    *ECONOMIC_CONTEXT_COLUMNS,
]

ECONOMIC_LEAKAGE_COLUMNS = [
    *ECONOMIC_TARGET_COLUMNS,
    *ECONOMIC_CONTEXT_COLUMNS,
]
```

Use these constants in `get_economic_targets()` instead of repeating the output column list.

Acceptance criteria:

- Economic target column names have one canonical definition in core.
- `get_economic_targets()` output order remains explicit and tested.

### 2. Extract Pure Economic Derivation Helper

Add a pure DataFrame helper in `fof8_core.targets.economic`:

```python
def add_economic_derived_columns(
    df: pl.DataFrame,
    merit_threshold: float = 0,
) -> pl.DataFrame:
    ...
```

This helper should assume the input contains:

- `Peak_Overall`
- `Career_Merit_Cap_Share`
- `Career_Games_Played`

It should:

- fill null source values
- derive `DPO`
- derive `Cleared_Sieve`
- derive `Economic_Success`
- derive `Positive_Career_Merit_Cap_Share`
- derive `Positive_DPO`

Then update `get_economic_targets()` so it only:

1. loads source target frames
2. unions IDs
3. joins source columns
4. calls `add_economic_derived_columns(...)`
5. selects `ECONOMIC_OUTPUT_COLUMNS`

Acceptance criteria:

- Economic formulas exist in exactly one helper.
- Unit tests cover the helper directly, including negative merit, null source values, and
  nonzero `merit_threshold`.

### 3. Replace ML Hard-Coded Target Lists

Edit `fof8-ml/src/fof8_ml/data/economic_dataset.py`.

Import target constants from core:

```python
from fof8_core.targets.economic import ECONOMIC_TARGET_COLUMNS, get_economic_targets
```

Replace the local `target_columns = [...]` list with:

```python
target_columns = ECONOMIC_TARGET_COLUMNS
```

Also consider filling all target columns with a small shared loop or helper instead of listing
each derived column manually. Keep this file focused on dataset assembly, not semantic formulas.

Acceptance criteria:

- `economic_dataset.py` no longer owns target column names or target formulas.
- Adding/removing an economic target requires changing core constants and tests, not ML dataset
  assembly code.

### 4. Remove Runtime Migration Derivation

Edit `fof8-ml/src/fof8_ml/orchestration/data_loader.py`.

Remove the migration block that silently creates missing economic target columns:

- `Positive_Career_Merit_Cap_Share`
- `Positive_DPO`
- `Economic_Success`
- `Cleared_Sieve`

Replace it with strict schema validation:

```python
from fof8_core.targets.economic import ECONOMIC_LEAKAGE_COLUMNS

missing_target_cols = sorted(c for c in ECONOMIC_LEAKAGE_COLUMNS if c not in df.columns)
if missing_target_cols:
    raise ValueError(
        "Processed features are missing economic target columns "
        f"{missing_target_cols}. Re-run feature processing."
    )
```

Do not preserve automatic fallback derivation for older parquet files. This makes stale data fail
fast and keeps formula ownership in core.

Acceptance criteria:

- `data_loader.py` does not duplicate economic target formulas.
- Old processed feature stores with missing target columns fail with a clear regenerate-data
  error.

### 5. Make Target Registry Parameterized

The current registry type is:

```python
TargetBuilder = Callable[[FOF8Loader], pl.DataFrame]
```

Change it to support keyword parameters:

```python
TargetBuilder = Callable[..., pl.DataFrame]

def get_target(name: str, loader: FOF8Loader, **kwargs: object) -> pl.DataFrame:
    ...
    return target_fn(loader, **kwargs)
```

Register `get_economic_targets` normally. Then callers can use:

```python
get_target("economic_targets", loader, merit_threshold=0.2)
```

Update tests to verify:

- existing no-kwargs target lookup still works
- `economic_targets` accepts `merit_threshold`
- unknown target errors still list available target names

Acceptance criteria:

- Target builders can expose meaningful parameters without special wrappers.
- The default `get_target(name, loader)` call remains simple for parameterless targets.

### 6. Move Leakage Prevention Toward Target Contracts

Update `pipelines/conf/target/economic.yaml`.

Prefer target identity and selected columns over a duplicated leakage column list:

```yaml
target_name: "economic_targets"
classifier_sieve:
  target_col: "Economic_Success"
  merit_threshold: 0
  min_positive_recall: 0.95
regressor_intensity:
  target_col: "Positive_Career_Merit_Cap_Share"
  target_space: "raw"
  target_family: "economic"
```

Then update `data_loader.py` to derive leakage columns from the target contract:

```python
target_cols = [
    cfg.target.classifier_sieve.target_col,
    cfg.target.regressor_intensity.target_col,
    *ECONOMIC_LEAKAGE_COLUMNS,
]
target_cols = list(dict.fromkeys(target_cols))
```

For now, this can be economic-specific. A later generalization can introduce a formal
`TargetSpec` object if multiple target families need the same behavior.

Acceptance criteria:

- `pipelines/conf/target/economic.yaml` no longer repeats all target/leakage columns.
- Leakage prevention remains explicit in code through imported core constants.

### 7. Decide What to Do With `final_sim_year`

`build_economic_dataset()` currently accepts `final_sim_year` and discards it.

Prefer one of these clean options:

1. Remove the argument if all callers can be updated in the same change.
2. If public API churn is too high, keep it temporarily but mark it as deprecated in the
   docstring and open a follow-up removal task.

Since this plan prioritizes architecture over compatibility, option 1 is preferred.

Update:

- `fof8-ml/src/fof8_ml/data/economic_dataset.py`
- `pipelines/process_features.py`
- tests that call `build_economic_dataset(...)`
- notebooks only if they are maintained as executable examples

Acceptance criteria:

- The dataset builder signature only contains parameters that affect behavior.

### 8. Test Coverage

Update or add tests:

```text
fof8-core/tests/test_economic_targets.py
fof8-core/tests/test_targets_registry.py
fof8-ml/tests/test_dataset.py
fof8-ml/tests/test_data_loader_cache.py
```

Required coverage:

- `ECONOMIC_OUTPUT_COLUMNS` matches `get_economic_targets()` output order.
- `add_economic_derived_columns()` handles null, negative, zero, and positive merit values.
- Parameterized registry lookup passes `merit_threshold`.
- ML dataset builder uses core constants and still excludes all economic targets from `X`.
- Data loader fails clearly when processed features are missing economic target columns.
- Config no longer needs `leakage_prevention.drop_cols`.

### 9. Verification

Run focused tests:

```bash
pytest fof8-core/tests/test_economic_targets.py fof8-core/tests/test_targets_registry.py
pytest fof8-ml/tests/test_dataset.py fof8-ml/tests/test_data_loader_cache.py
```

Then run the broader package suites if time allows:

```bash
pytest fof8-core/tests
pytest fof8-ml/tests
```

## Non-Goals

- Do not add a full generic `TargetSpec` abstraction unless another target family needs it now.
- Do not keep automatic derivation for stale processed feature stores.
- Do not make notebooks the source of truth for target definitions.
- Do not refactor unrelated target families beyond the registry signature needed for
  parameterized target builders.

## Final Acceptance Criteria

- Economic target formulas and column names have one source of truth in `fof8_core.targets`.
- `fof8_ml` has no duplicated economic target derivation formulas.
- Target registry supports parameterized target builders.
- Economic config does not duplicate leakage target columns.
- Stale processed data fails fast with a clear reprocessing error.
- Focused core and ML tests pass.
