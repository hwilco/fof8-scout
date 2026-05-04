# Target Family Split Plan

## Goal

Separate scouted talent outcomes, economic outcomes, and hybrid/composite value outcomes into
clear target families. `Peak_Overall` should not live in economic targets, and `DPO` should be
explicitly modeled as a composite target because it combines scouted ratings with economic value.

Prefer clear architecture over preserving the current `economic_targets` shape.

## Current Problem

`fof8_core.targets.economic` currently bundles three concepts:

- Talent outcome: `Peak_Overall`
- Economic outcomes: `Career_Merit_Cap_Share`, `Positive_Career_Merit_Cap_Share`,
  `Economic_Success`, `Cleared_Sieve`
- Composite outcomes: `DPO`, `Positive_DPO`

This makes the module name misleading and causes downstream code to treat rating-derived and
economic-derived labels as one target family.

## Desired End State

- Talent targets live in `fof8_core.targets.talent`.
- Economic targets live in `fof8_core.targets.economic`.
- Composite targets live in `fof8_core.targets.composite`.
- A higher-level draft outcome bundle composes target families for ML dataset generation.
- Leakage-prevention column lists are derived from target-family constants, not duplicated in
  config.
- `DPO` remains available as a baseline target, but its hybrid nature is obvious from the module
  boundary.

## Proposed Modules

### `fof8_core.targets.talent`

Owns scouted/rating-based outcome targets.

Move `get_peak_overall()` out of `fof8_core.targets.financial` into this module.

Suggested constants:

```python
TALENT_TARGET_COLUMNS = [
    "Peak_Overall",
]

TALENT_OUTPUT_COLUMNS = [
    "Player_ID",
    *TALENT_TARGET_COLUMNS,
]
```

Implementation:

```python
def get_peak_overall(loader: FOF8Loader, k: int = 3) -> pl.DataFrame:
    ...
```

Keep behavior identical to the current implementation.

### `fof8_core.targets.economic`

Owns market/money outcome targets only.

Suggested constants:

```python
ECONOMIC_TARGET_COLUMNS = [
    "Cleared_Sieve",
    "Economic_Success",
    "Career_Merit_Cap_Share",
    "Positive_Career_Merit_Cap_Share",
]

ECONOMIC_SOURCE_COLUMNS = [
    "Career_Merit_Cap_Share",
]

ECONOMIC_OUTPUT_COLUMNS = [
    "Player_ID",
    *ECONOMIC_TARGET_COLUMNS,
]
```

Implementation:

```python
def add_economic_derived_columns(
    df: pl.DataFrame,
    merit_threshold: float = 0,
) -> pl.DataFrame:
    ...

def get_economic_targets(
    loader: FOF8Loader,
    merit_threshold: float = 0,
) -> pl.DataFrame:
    ...
```

`get_economic_targets()` should call `get_merit_cap_share(loader)`, derive economic-only labels,
and select `ECONOMIC_OUTPUT_COLUMNS`.

Do not import or call `get_peak_overall()` from this module.

### `fof8_core.targets.composite`

Owns labels that intentionally combine multiple target families.

Suggested constants:

```python
COMPOSITE_TARGET_COLUMNS = [
    "DPO",
    "Positive_DPO",
]

COMPOSITE_SOURCE_COLUMNS = [
    "Peak_Overall",
    "Career_Merit_Cap_Share",
]

COMPOSITE_OUTPUT_COLUMNS = [
    "Player_ID",
    *COMPOSITE_TARGET_COLUMNS,
]
```

Implementation:

```python
def add_dpo_columns(df: pl.DataFrame) -> pl.DataFrame:
    ...

def get_dpo_targets(loader: FOF8Loader) -> pl.DataFrame:
    ...
```

`get_dpo_targets()` should compose:

- `get_peak_overall(loader)` from `talent.py`
- `get_merit_cap_share(loader)` from `financial.py` or the economic source function

Then derive:

```text
DPO = Peak_Overall * Career_Merit_Cap_Share
Positive_DPO = max(DPO, 0)
```

### `fof8_core.targets.draft_outcomes`

Owns the convenient broad target bundle used by ML dataset generation.

Suggested constants:

```python
DRAFT_OUTCOME_TARGET_COLUMNS = [
    *ECONOMIC_TARGET_COLUMNS,
    *COMPOSITE_TARGET_COLUMNS,
]

DRAFT_OUTCOME_CONTEXT_COLUMNS = [
    *TALENT_TARGET_COLUMNS,
    "Career_Games_Played",
]

DRAFT_OUTCOME_LEAKAGE_COLUMNS = [
    *DRAFT_OUTCOME_TARGET_COLUMNS,
    *DRAFT_OUTCOME_CONTEXT_COLUMNS,
]

DRAFT_OUTCOME_OUTPUT_COLUMNS = [
    "Player_ID",
    *DRAFT_OUTCOME_TARGET_COLUMNS,
    *DRAFT_OUTCOME_CONTEXT_COLUMNS,
]
```

Implementation:

```python
def get_draft_outcome_targets(
    loader: FOF8Loader,
    merit_threshold: float = 0,
) -> pl.DataFrame:
    ...
```

This function should compose:

- `get_economic_targets(loader, merit_threshold=merit_threshold)`
- `get_dpo_targets(loader)`
- `get_peak_overall(loader)`
- `get_career_outcomes(loader)` for `Career_Games_Played`

It should join on the union of all `Player_ID`s needed by the component target frames, fill
missing ML-facing target values to zero, and select `DRAFT_OUTCOME_OUTPUT_COLUMNS`.

This replaces the current broad use of `get_economic_targets()` in ML dataset assembly.

## Step-by-Step Implementation

### 1. Create `talent.py`

Add:

```text
fof8-core/src/fof8_core/targets/talent.py
```

Move `get_peak_overall()` from `financial.py` into `talent.py`.

Update imports:

- `fof8-core/src/fof8_core/targets/__init__.py`
- `fof8-core/src/fof8_core/targets/registry.py`
- any module currently importing `get_peak_overall` from `financial.py`

Because this plan prefers clean architecture, remove the old export from `financial.py` rather
than keeping a compatibility re-export.

### 2. Narrow `economic.py`

Edit:

```text
fof8-core/src/fof8_core/targets/economic.py
```

Remove:

- `Peak_Overall`
- `DPO`
- `Positive_DPO`
- `Career_Games_Played`

Keep:

- `Career_Merit_Cap_Share`
- `Positive_Career_Merit_Cap_Share`
- `Economic_Success`
- `Cleared_Sieve`

Ensure `add_economic_derived_columns()` no longer needs `Peak_Overall` or
`Career_Games_Played`.

### 3. Add `composite.py`

Add:

```text
fof8-core/src/fof8_core/targets/composite.py
```

Implement `add_dpo_columns()` and `get_dpo_targets()`.

Keep all DPO math here. Do not duplicate it in `economic.py` or `fof8_ml`.

### 4. Add `draft_outcomes.py`

Add:

```text
fof8-core/src/fof8_core/targets/draft_outcomes.py
```

Implement `get_draft_outcome_targets()` and the draft-outcome constants.

This is the only target builder that should intentionally return the broad ML-ready bundle:

- classifier targets
- regressor targets
- comparison/baseline targets
- leakage/context columns needed to drop from features

### 5. Update Registry

Edit:

```text
fof8-core/src/fof8_core/targets/registry.py
```

Register:

```python
TARGET_REGISTRY["peak_overall"] = get_peak_overall
TARGET_REGISTRY["economic_targets"] = get_economic_targets
TARGET_REGISTRY["dpo_targets"] = get_dpo_targets
TARGET_REGISTRY["draft_outcome_targets"] = get_draft_outcome_targets
```

Remove `get_peak_overall` imports from `financial.py`.

If the registry already supports keyword parameters, keep using that. If not, update it so:

```python
def get_target(name: str, loader: FOF8Loader, **kwargs: object) -> pl.DataFrame:
    ...
```

works for `merit_threshold`.

### 6. Update Public Exports

Edit:

```text
fof8-core/src/fof8_core/targets/__init__.py
```

Export:

- `get_peak_overall`
- `get_economic_targets`
- `get_dpo_targets`
- `get_draft_outcome_targets`
- relevant column constants if they are intended for ML consumers

Do not export `get_peak_overall` from `financial.py`.

### 7. Update ML Dataset Builder

Edit:

```text
fof8-ml/src/fof8_ml/data/economic_dataset.py
```

Replace:

```python
from fof8_core.targets.economic import ECONOMIC_TARGET_COLUMNS, get_economic_targets
```

with:

```python
from fof8_core.targets.draft_outcomes import (
    DRAFT_OUTCOME_TARGET_COLUMNS,
    get_draft_outcome_targets,
)
```

Replace:

```python
df_targets = get_economic_targets(loader, merit_threshold=merit_threshold)
```

with:

```python
df_targets = get_draft_outcome_targets(loader, merit_threshold=merit_threshold)
```

Replace local target-column usage with:

```python
target_columns = DRAFT_OUTCOME_TARGET_COLUMNS
```

If `Peak_Overall` and `Career_Games_Played` are still included in the joined frame, ensure they
are excluded from model features through leakage/context columns in the downstream loader or by
dropping them in this builder when producing `X`.

### 8. Update Data Loader Leakage Handling

Edit:

```text
fof8-ml/src/fof8_ml/orchestration/data_loader.py
```

Use:

```python
from fof8_core.targets.draft_outcomes import DRAFT_OUTCOME_LEAKAGE_COLUMNS
```

Build `target_cols` from:

- configured classifier target
- configured regressor target
- `DRAFT_OUTCOME_LEAKAGE_COLUMNS`

Remove any hard-coded economic/DPO/Peak column lists in ML code.

### 9. Update Config

Edit:

```text
pipelines/conf/target/economic.yaml
```

Prefer:

```yaml
target_name: "draft_outcome_targets"
classifier_sieve:
  target_col: "Economic_Success"
  merit_threshold: 0
  min_positive_recall: 0.95
regressor_intensity:
  target_col: "Positive_Career_Merit_Cap_Share"
  target_space: "raw"
  target_family: "economic"
```

Remove duplicated `leakage_prevention.drop_cols`.

### 10. Update Tests

Add or update tests:

```text
fof8-core/tests/test_talent_targets.py
fof8-core/tests/test_economic_targets.py
fof8-core/tests/test_composite_targets.py
fof8-core/tests/test_draft_outcome_targets.py
fof8-core/tests/test_targets_registry.py
fof8-ml/tests/test_dataset.py
fof8-ml/tests/test_data_loader_cache.py
```

Required coverage:

- `get_peak_overall()` works from `talent.py`.
- `financial.py` no longer owns or exports `get_peak_overall`.
- `get_economic_targets()` does not include `Peak_Overall`, `DPO`, `Positive_DPO`, or
  `Career_Games_Played`.
- `get_dpo_targets()` derives `DPO` and `Positive_DPO` correctly.
- `get_draft_outcome_targets()` returns the broad ML bundle in stable column order.
- Registry lookup works for all target families.
- ML dataset generation still returns expected `X`, `y`, and metadata.
- Leakage/context columns are excluded from model features.

### 11. Update Docs

Update docs that currently describe DPO as an economic target:

- `docs/draft_value_target_strategy.md`
- `docs/draft_model_finish_line_plan.md`
- `docs/economic_targets_architecture_cleanup_plan.md`, if still relevant

Recommended wording:

- `Peak_Overall` is a talent/rating target.
- `Career_Merit_Cap_Share` is an economic target.
- `DPO` is a composite target and should be kept as a baseline/comparison target unless
  experiments justify making it primary.

## Verification

Run focused tests:

```bash
pytest fof8-core/tests/test_talent_targets.py \
       fof8-core/tests/test_economic_targets.py \
       fof8-core/tests/test_composite_targets.py \
       fof8-core/tests/test_draft_outcome_targets.py \
       fof8-core/tests/test_targets_registry.py

pytest fof8-ml/tests/test_dataset.py fof8-ml/tests/test_data_loader_cache.py
```

Then run broader suites:

```bash
pytest fof8-core/tests
pytest fof8-ml/tests
```

## Non-Goals

- Do not redesign the model pipeline.
- Do not remove DPO as an available target.
- Do not calibrate or replace `Career_Merit_Cap_Share`.
- Do not introduce a generic target abstraction beyond the family constants and broad
  draft-outcome bundle unless the implementation becomes repetitive.

## Final Acceptance Criteria

- `Peak_Overall` lives under talent targets, not financial or economic targets.
- Economic targets no longer include scouted/rating outputs or DPO.
- DPO lives under composite targets and explicitly depends on talent plus economic source data.
- ML dataset generation uses `draft_outcome_targets` for the broad training bundle.
- Leakage prevention derives from draft-outcome constants, not duplicated YAML lists.
- Tests pass across core target families and ML dataset loading.
