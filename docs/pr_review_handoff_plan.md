# PR #1 Review Feedback Handoff Plan

## Scope
Address all unresolved review threads on PR #1 (`refactor/modular-ml-pipeline` -> `main`) with code fixes, tests, and thread responses.

## Source Threads
- `#discussion_r3178328352` (P1): guard stage-2 config dereference in `sweep_manager.py`
- `#discussion_r3178328356` (P2): stage run naming contract with `batch_inference.py`
- `#discussion_r3178336959` (critical): duplicate of missing config key dereference in `sweep_manager.py`
- `#discussion_r3178336965` (high): champion params still reference legacy keys in `sweep_manager.py`
- `#discussion_r3178336962` (high): CatBoost regressor forces RMSE in `catboost_wrapper.py`
- `#discussion_r3178336967` (medium): CatBoost GPU `devices="0"` hardcoded in `catboost_wrapper.py`
- `#discussion_r3178336966` (medium): loader should leverage `schema_override` during CSV scan
- `#discussion_r3178336971` (medium): silent fallback threshold behavior in `evaluator.py`

## Agent Split (Disjoint Ownership)

### Agent A: Sweep + Inference Contract (highest risk)
**Own files**
- `fof8-ml/src/fof8_ml/orchestration/sweep_manager.py`
- `fof8-ml/src/fof8_ml/orchestration/experiment_logger.py`
- `pipelines/batch_inference.py`
- New/updated tests under `fof8-ml/tests/` and/or `pipelines/tests/`

**Tasks**
1. Replace unsafe direct config attribute access in `update_champion` with compatibility-safe logic.
2. Support both refactored and legacy config shapes for champion tagging and model registration.
3. Normalize stage nested run names with a shared naming contract used by both training and inference.
4. Update inference lookup to resolve models from the canonical stage names (and optionally legacy aliases).

**Acceptance criteria**
- Classifier-only and regressor-only runs do not raise missing-key errors.
- Champion tags/notes contain non-empty model parameter summaries.
- Parent-run model discovery succeeds with current stage run naming.
- Both duplicate sweep comments are fully addressed by one coherent fix.

**Thread closure target**
- `r3178328352`, `r3178336959`, `r3178336965`, `r3178328356`

---

### Agent B: CatBoost Wrapper Flexibility
**Own files**
- `fof8-ml/src/fof8_ml/models/catboost_wrapper.py`
- Related tests in `fof8-ml/tests/`

**Tasks**
1. Change regressor objective behavior from forced overwrite to default-only (allow config override).
2. Remove hardcoded GPU `devices="0"` default behavior; only set when explicitly configured.
3. Ensure existing GPU/non-GPU behavior remains stable.

**Acceptance criteria**
- Regressor defaults to RMSE only when no objective is provided.
- `Tweedie`/`Poisson` objective configuration is preserved end-to-end.
- GPU device selection is user-configurable and not forcibly pinned to device 0.

**Thread closure target**
- `r3178336962`, `r3178336967`

---

### Agent C: Loader + Evaluator Robustness
**Own files**
- `fof8-core/src/fof8_core/loader.py`
- `fof8-ml/src/fof8_ml/orchestration/evaluator.py`
- Related tests in `fof8-core/tests/` and `fof8-ml/tests/`

**Tasks**
1. Push applicable schema overrides into `pl.scan_csv(..., schema_overrides=...)` at read time.
2. Keep only necessary post-read casts for columns that must remain deferred.
3. Improve threshold optimization behavior when `min_positive_recall` is infeasible:
   - emit warning/flag
   - return deterministic fallback behavior with explicit signaling

**Acceptance criteria**
- Loader respects standardized schema overrides during parsing for supported fields.
- No regression in mixed-type injury columns or downstream transforms.
- Infeasible threshold constraints are observable (warning or explicit status) and tested.

**Thread closure target**
- `r3178336966`, `r3178336971`

## Integration + Verification

## Merge order
1. Agent A (contract/risk blockers)
2. Agent B (model wrapper config correctness)
3. Agent C (robustness/perf)

## Required test pass set
- Targeted unit tests added for each changed module.
- Existing relevant suites:
  - `fof8-ml/tests/test_model_factory.py`
  - `fof8-ml/tests/test_data_loader_cache.py`
  - `fof8-core/tests/test_loader.py`
- One end-to-end smoke:
  - classifier pipeline run
  - regressor pipeline run
  - model resolution via `pipelines/batch_inference.py`

## PR Thread response template (per comment)
1. State root cause briefly.
2. Link exact fix location(s).
3. State tests added/updated.
4. Confirm behavior before/after.
5. Resolve thread.

## Definition of Done
- All 8 unresolved threads resolved (with 2 sweep duplicates handled by shared fix).
- CI green for touched areas.
- No regression in pipeline model registration/discovery.
- PR description updated with a short “review feedback addressed” addendum summarizing final changes.
