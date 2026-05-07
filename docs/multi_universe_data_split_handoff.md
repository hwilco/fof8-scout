# Multi-Universe Data Loading And Split Strategy Handoff

## Goal

Support training from multiple raw simulation folders under `fof8-gen/data/raw/`, with train/test splitting configurable as either:

- `chronological`: hold out the latest eligible draft classes within each universe.
- `random`: split across the pooled eligible rows or draft classes using a deterministic seed.

The end state should let one run pool universes such as `DRAFT003`, `DRAFT004`, and `DRAFT005` into one processed feature store, then train classifier/regressor models against either split strategy.

## Current State

The pipeline is single-universe today.

- `pipelines/conf/data/fof8_base.yaml` has one `league_name`.
- `pipelines/process_features.py` builds one `FOF8Loader`, discovers that one league's years, builds one parquet, and writes `fof8-ml/data/processed/features.parquet`.
- `fof8-core/src/fof8_core/loader.py` models one `league_dir`.
- `fof8-ml/src/fof8_ml/orchestration/data_loader.py` reads the processed parquet, discovers one timeline through `FOF8Loader`, then performs a hard-coded chronological split.
- Feature/target joins commonly use only `Player_ID`, which is unsafe once multiple universes are pooled.

## Main Design Decision

Keep `FOF8Loader` single-universe and add pooling above it.

This preserves existing loader behavior and limits the change to feature materialization, processed schema, and training split orchestration. Each universe should be processed independently with the existing loader and target builders, tagged with a stable universe identifier, then concatenated.

## Required Processed Schema Change

Add a stable universe identity column to processed rows, for example:

```text
Universe
```

The value should match the raw folder name by default, such as `DRAFT005`.

This column must be treated as metadata, not a model feature, unless explicitly added later as a deliberate feature experiment.

Recommended metadata columns:

```python
["Universe", "Player_ID", "Year", "First_Name", "Last_Name"]
```

## Code Changes

### Feature Materialization

Update `pipelines/process_features.py` to:

1. Resolve selected universes from config.
2. For each universe:
   - create `FOF8Loader(base_path=absolute_raw_path, league_name=universe)`
   - discover `initial_sim_year` and `final_sim_year`
   - build that universe's full valid year range
   - call `build_economic_dataset(...)`
   - add `Universe`
3. Concatenate all universe frames.
4. Write the pooled `features.parquet`.

Support both explicit and discoverable config shapes if practical:

```yaml
league_names: ["DRAFT003", "DRAFT004", "DRAFT005"]
# optional alternative:
league_glob: "DRAFT*"
```

Retain backward compatibility with existing `league_name` if possible.

### Dataset Builders

Update `fof8-ml/src/fof8_ml/data/economic_dataset.py` so metadata can include `Universe`.

The safest short-term approach is to keep each call single-universe:

- Build targets and features with existing functions.
- Join within that universe as today.
- Add `Universe` after the join, before returning metadata.

Avoid pooling before joins unless every target builder has been made universe-aware.

### Training Data Loader

Update `fof8-ml/src/fof8_ml/orchestration/data_loader.py` to derive timelines from the processed parquet instead of from a single `FOF8Loader`.

Expected behavior:

- Validate `Universe` exists when multiple universes are configured.
- Apply right-censoring per universe:
  - `valid_start_year = min(Year) per universe`
  - `valid_end_year = max(Year) - cfg.split.right_censor_buffer` per universe
- Filter out rows later than each universe's valid end year.
- Branch on `cfg.split.strategy`.

For `chronological`:

- Compute train/test cutoff independently per universe.
- Pool all per-universe train rows into `train_df`.
- Pool all per-universe test rows into `test_df`.

For `random`:

- Use a deterministic seed from `cfg.split.seed` or fallback to global `cfg.seed`.
- Decide split unit:
  - `row`: randomly split eligible rows.
  - `draft_class`: randomly split `(Universe, Year)` groups, preserving whole draft classes.
- Prefer `draft_class` if model evaluation should remain draft-board oriented.
- Stratify or at least balance by `Universe` so small universes are not accidentally absent from holdout.

### Pipeline Types And Summary

`TimelineInfo` currently stores one global range. For pooled data, either extend it or add a new structure that records:

- included universes
- per-universe initial/final/valid/train/test ranges
- split strategy
- split unit for random splits
- train/test row counts

Update `DataLoader.print_summary()` to report pooled data clearly.

### Cache Hashes

Include these fields in `DataLoader.load()` cache hashing:

- selected universe list or glob
- split strategy
- split seed
- split unit
- right-censor buffer
- test split percentage

Also review `fof8-ml/src/fof8_ml/data/cache.py`; it currently keys raw data identity around one `league_name`.

### Batch Inference

`pipelines/batch_inference.py` still loads one configured league. It can remain single-universe initially, but document that pooled training does not automatically imply pooled inference. If batch inference should run across universes too, mirror the same universe config resolution there.

## Config Changes

Update `pipelines/conf/data/fof8_base.yaml`:

```yaml
raw_path: "fof8-gen/data/raw"
league_names: ["DRAFT005"]
processed_dir: "fof8-ml/data/processed"
features_path: "fof8-ml/data/processed/features.parquet"
active_team_id: null
```

Add `pipelines/conf/split/random.yaml`:

```yaml
strategy: "random"
test_split_pct: 0.20
right_censor_buffer: 20
seed: 42
unit: "draft_class"
stratify_by: ["Universe"]
```

Keep `pipelines/conf/split/chronological.yaml`, but consider making strategy-specific fields explicit:

```yaml
strategy: "chronological"
test_split_pct: 0.20
right_censor_buffer: 20
```

The classifier and regressor pipeline defaults can continue to use:

```yaml
- split: chronological
```

Users can override with:

```bash
uv run python pipelines/train_classifier.py split=random
uv run python pipelines/train_regressor.py split=random
```

## Documentation Changes

Update:

- `README.md`
  - raw data can contain multiple universe folders
  - `dvc repro` can build a pooled feature store
  - train/test split can be chronological or random

- `fof8-ml/README.md`
  - replace any "chronological splitting in-memory" only language
  - document split override examples

- `pipelines/conf/README.md`
  - document `data.league_names`
  - document `split.strategy`, `split.unit`, and `split.seed`

## Test Plan

Add focused unit tests before wiring broad integration tests.

### Core/Feature Tests

- Single-universe behavior remains unchanged.
- Processing multiple universes preserves `Universe`.
- Duplicate `Player_ID` values across universes do not collide after materialization.

### Data Loader Tests

- Chronological split is computed per universe.
- Random row split is deterministic for a fixed seed.
- Random draft-class split keeps all rows for a `(Universe, Year)` in the same side.
- Cache hash changes for:
  - universe list
  - split strategy
  - split seed
  - split unit
  - test percentage
  - right-censor buffer
- `Universe` is excluded from feature columns by default.

### Config Tests

- Existing `league_name` config still works if backward compatibility is kept.
- `league_names` config with multiple folders resolves correctly.
- `split=random` Hydra override resolves without missing keys.

## Implementation Order

1. Add config parsing helpers for universe selection.
2. Update `process_features.py` to pool independently built universes and add `Universe`.
3. Update metadata handling so `Universe` is excluded from features.
4. Refactor `DataLoader.load()` split logic into small helpers:
   - valid range filtering
   - chronological split
   - random split
5. Extend timeline/summary structures.
6. Update cache hashes.
7. Add tests.
8. Update README and config docs.
9. Run targeted tests, then full affected suites.

## Open Questions

- Should random splitting operate by row or by draft class by default? Draft class is better aligned with draft-board evaluation; row split gives more uniform target distribution.
- Should random splits stratify by classifier target as well as `Universe`? This may be worthwhile if positive cases are sparse.
- Should `Universe` ever become an optional feature? Default should be no, to avoid learning simulation-specific artifacts.
- Should `features.parquet` remain one pooled file, or should processed output include per-universe parquet shards? One pooled file is simpler for current training code.

## Verification Commands

Suggested commands after implementation:

```bash
uv run pytest fof8-core/tests/test_loader.py fof8-ml/tests/test_dataset.py fof8-ml/tests/test_data_loader_cache.py
uv run python pipelines/process_features.py
uv run python pipelines/train_classifier.py split=chronological
uv run python pipelines/train_classifier.py split=random
uv run python pipelines/train_regressor.py split=random
```

If the full pipeline is expensive, run the unit tests first and use a small two-universe fixture or temporary processed parquet for split tests.
