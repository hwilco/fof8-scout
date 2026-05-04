# Economic Targets Refactor Handoff

## Goal

Move canonical economic target construction out of `fof8_ml.data.economic_dataset`
and into `fof8_core.targets`, while keeping ML dataset assembly in `fof8_ml`.

## Implementation Plan

### 1. Add `fof8_core.targets.economic`

Create `fof8-core/src/fof8_core/targets/economic.py`.

Implement:

```python
def get_economic_targets(loader: FOF8Loader, merit_threshold: float = 0) -> pl.DataFrame:
    ...
```

Move the logic currently in `build_economic_dataset()` that:

- calls `get_peak_overall(loader)`
- calls `get_merit_cap_share(loader)`
- calls `get_career_outcomes(loader)`
- unions all `Player_ID`s
- joins `Peak_Overall`, `Career_Merit_Cap_Share`, and `Career_Games_Played`
- fills null source values
- derives:
  - `DPO`
  - `Cleared_Sieve`
  - `Economic_Success`
  - `Positive_Career_Merit_Cap_Share`
  - `Positive_DPO`
- selects the final target columns

Keep the output schema identical to the current `df_targets` in
`fof8-ml/src/fof8_ml/data/economic_dataset.py`.

### 2. Update Target Exports

Edit `fof8-core/src/fof8_core/targets/__init__.py`:

- import `get_economic_targets`
- add it to `__all__`

Edit `fof8-core/src/fof8_core/targets/registry.py`:

- import/register `get_economic_targets`
- add registry key `"economic_targets"`

The registry currently accepts only `Callable[[FOF8Loader], pl.DataFrame]`, so register
the default-threshold version only:

```python
TARGET_REGISTRY["economic_targets"] = get_economic_targets
```

Do not make the registry config-aware in this refactor.

### 3. Simplify `economic_dataset.py`

Edit `fof8-ml/src/fof8_ml/data/economic_dataset.py`.

Replace these imports:

```python
from fof8_core.targets.career import get_career_outcomes
from fof8_core.targets.financial import get_merit_cap_share, get_peak_overall
```

with:

```python
from fof8_core.targets.economic import get_economic_targets
```

Replace the `df_peak` / `df_merit` / `df_outcomes` / `all_ids` / `df_targets`
block with:

```python
df_targets = get_economic_targets(loader, merit_threshold=merit_threshold)
```

Leave all feature assembly, joins, duplicate-column checks, null-fill after feature join,
categorical handling, and X/y/metadata splitting in `economic_dataset.py`.

### 4. Keep Runtime Migration Logic

Do not remove the migration safety block in
`fof8-ml/src/fof8_ml/orchestration/data_loader.py` in this refactor. It supports older
processed parquet files.

Optional small cleanup: add a comment there pointing to
`fof8_core.targets.economic.get_economic_targets` as the canonical source for newly
processed data.

### 5. Add Core Tests

Add or extend tests in `fof8-core/tests`.

Suggested test file:

```text
fof8-core/tests/test_economic_targets.py
```

Test cases:

- `get_economic_targets(mock_loader)` returns the expected columns.
- Undrafted or missing-merit players are preserved via the union of IDs.
- Null `Peak_Overall`, `Career_Merit_Cap_Share`, and `Career_Games_Played` are filled to
  zero.
- `DPO = Peak_Overall * Career_Merit_Cap_Share`.
- `Positive_DPO` and `Positive_Career_Merit_Cap_Share` clip negatives to zero.
- `Economic_Success` is `Career_Merit_Cap_Share > 0`.
- `Cleared_Sieve` respects a nonzero `merit_threshold`.

Also update `fof8-core/tests/test_targets_registry.py` to expect
`"economic_targets"` in the built-ins.

### 6. Keep ML Tests Mostly Unchanged

Existing `fof8-ml/tests/test_dataset.py::test_build_economic_dataset_preserves_undrafted_peak`
should still pass. It becomes an integration check proving `build_economic_dataset()` still
returns the same `y` and metadata after delegation.

If mocks patch old function imports directly, update them to patch either:

```python
fof8_core.targets.economic.get_economic_targets
```

or rely on the existing `mock_loader`.

### 7. Run Focused Verification

From the repo root, run:

```bash
pytest fof8-core/tests/test_targets_registry.py fof8-core/tests/test_economic_targets.py
pytest fof8-ml/tests/test_dataset.py
```

If the monorepo uses package-specific environments, run the equivalent project commands
already used in this repo.

## Acceptance Criteria

- `build_economic_dataset()` output is behaviorally unchanged.
- Economic target semantics live in `fof8_core.targets`.
- `fof8_ml.data.economic_dataset` only assembles ML-ready feature, target, and metadata
  frames.
- `get_target("economic_targets", loader)` works with default `merit_threshold=0`.
- Existing processed-parquet compatibility in `data_loader.py` remains intact.
