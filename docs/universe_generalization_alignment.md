# Universe Generalization Alignment

## Context

The target use case is not "predict future draft classes in the same simulation." The target
use case is "predict a draft class from a new universe that the model has never seen before."

With 15 universes and roughly 30 eligible draft years per universe after right censoring,
the evaluation strategy should optimize for **generalization across universes**, not for
interpolation within the same universe.

This note reviews the current pipeline and lists the changes needed to better align it with
that objective.

## Current State

### What already aligns

- The data loader already carries `Universe` metadata through training artifacts.
- Random splits can preserve whole draft classes using the `(Universe, Year)` grouping key.
- The regressor already uses `(Universe, Year)` metadata for draft-aware metrics.

Relevant code:

- `fof8-ml/src/fof8_ml/orchestration/data_loader.py`
- `fof8-ml/src/fof8_ml/orchestration/regressor.py`
- `pipelines/conf/split/random.yaml`

### What does not align yet

1. The train/test split does **not** support holding out entire universes.
   - `random` split samples rows or `(Universe, Year)` draft classes *within* each universe.
   - `chronological` split holds out the latest eligible years *within* each universe.

2. Cross-validation is still row-level.
   - Classifier CV uses `StratifiedKFold`.
   - Regressor CV uses `KFold`.
   - Neither respects `Universe` boundaries.

3. The explicit holdout set is carved out by `DataLoader`, but the training pipelines do not
   use it for model selection or final reporting.
   - The classifier and regressor optimize on OOF metrics from CV over `X_train`.
   - `X_test` / `meta_test` are loaded but not consumed by the training entrypoints.

Relevant code:

- `fof8-ml/src/fof8_ml/orchestration/trainer.py`
- `fof8-ml/src/fof8_ml/orchestration/classifier.py`
- `fof8-ml/src/fof8_ml/orchestration/regressor.py`

## Recommended Evaluation Philosophy

With 15 universes, the default workflow should be:

1. Hold out whole universes for validation and test.
2. Tune primarily against the validation universes.
3. Use grouped CV only inside the training universes when needed.
4. Keep the test universes completely untouched during model development.
5. Report final performance on unseen test universes only after the model design is frozen.

This answers the real production question:

> If a user starts a new universe, how well does the model generalize to that universe?

It also preserves a true final overfitting check:

> After tuning on training and validation universes, does performance still hold on universes
> the project never used for model selection?

## Recommended Changes

### 1. Add a universe-level split strategy

Add a new split mode that assigns entire universes to train / validation / test.

Suggested config shape:

```yaml
strategy: "grouped_universe"
right_censor_buffer: 20
train_universe_pct: 0.67
val_universe_pct: 0.13
test_universe_pct: 0.20
seed: 42
```

Suggested behavior:

- Apply the right-censor eligibility filter first.
- Sample universe ids, not rows.
- Keep every eligible year from a selected universe in the same split.

With 15 universes, a practical default is:

- 10 universes train
- 2 universes validation
- 3 universes test

or:

- 11 universes train
- 2 universes validation
- 2 universes test

Implementation targets:

- `fof8-ml/src/fof8_ml/orchestration/data_loader.py`
- `pipelines/conf/split/`
- `pipelines/conf/README.md`

### 2. Make validation explicit in `PreparedData`

The current `PreparedData` object only exposes:

- `X_train`
- `X_test`
- `meta_train`
- `meta_test`

That is too coarse for the intended workflow. Add an explicit validation split.

Suggested additions:

- `X_val`
- `y_cls_val`
- `y_reg_val`
- `meta_val`
- `outcomes_val` if cross-outcome reporting on validation matters

Implementation targets:

- `fof8-ml/src/fof8_ml/orchestration/pipeline_types.py`
- `fof8-ml/src/fof8_ml/orchestration/data_loader.py`

### 3. Stop using plain `KFold` / `StratifiedKFold` as the default

For this dataset, plain row-level CV answers the wrong question because the same universe can
appear in both train and validation folds.

Default replacements:

- Classifier: `StratifiedGroupKFold` if class balance is stable enough by universe
- Regressor: `GroupKFold`

Grouping key:

- `Universe` for the main unseen-universe objective

Secondary option for diagnostics only:

- `(Universe, Year)` grouped folds when you want a less coarse estimate than universe-level CV

Implementation targets:

- `fof8-ml/src/fof8_ml/orchestration/trainer.py`

Important constraint:

- The CV helpers currently accept only `X` and `y`.
- They will need `groups`, derived from `meta_train["Universe"]` or a passed array.

### 4. Separate "tuning mode" from "final reporting mode"

Right now the pipelines optimize directly on CV OOF metrics from the pooled training rows.
That is expensive and does not use the carved-out holdout set.

A better split of responsibilities:

#### Tuning mode

- Train on training universes
- Early stop on validation universes
- Optimize hyperparameters on validation metrics
- Fit calibration and choose decision thresholds on validation universes
- Use training subsampling if runtime becomes a problem

#### Final reporting mode

- Refit on train + validation universes with chosen hyperparameters
- Evaluate once on held-out test universes
- Log both validation and test metrics separately

Important rule:

- Test universes are not used for hyperparameter tuning
- Test universes are not used for threshold selection
- Test universes are not used for calibration fitting
- Test universes are not used for feature selection or model-family choice
- Once test results are used to make another modeling decision, they are no longer a true test set

Implementation targets:

- `fof8-ml/src/fof8_ml/orchestration/classifier.py`
- `fof8-ml/src/fof8_ml/orchestration/regressor.py`
- `pipelines/train_classifier.py`
- `pipelines/train_regressor.py`

### 5. Start using the explicit holdout set in metrics and artifacts

At the moment, `X_test` is loaded but not used by the training pipelines. That should change.

Minimum change:

- Score the final classifier and regressor on `X_test`
- Log holdout metrics under distinct names such as:
  - `classifier_test_pr_auc`
  - `classifier_test_hit_recall`
  - `regressor_test_rmse`
  - `regressor_test_draft_value_score`

Recommended additional artifacts:

- per-player test predictions
- per-universe test summaries
- per-universe calibration / ranking breakdowns

Implementation targets:

- `fof8-ml/src/fof8_ml/orchestration/experiment_logger.py`
- `fof8-ml/src/fof8_ml/orchestration/classifier.py`
- `fof8-ml/src/fof8_ml/orchestration/regressor.py`

### 6. Keep row-level random splits only for diagnostics

The existing `split=random unit=row` path can still be useful for:

- smoke tests
- feature debugging
- fast overfit checks

But it should not be the default reported metric for this project.

Recommended change:

- Mark row-level random split as diagnostic-only in config docs.
- Make universe-grouped split the default pipeline split once implemented.

Implementation targets:

- `pipelines/conf/classifier_pipeline.yaml`
- `pipelines/conf/regressor_pipeline.yaml`
- `pipelines/conf/README.md`

### 7. Add optional training-only subsampling for sweeps

With 15 universes, runtime will increase. The right place to control cost is the **training**
portion of the pipeline, not validation/test.

Recommended behavior:

- Never subsample validation or test universes by default.
- Optionally subsample only the training universes during broad sweeps.
- Refit finalists on the full training split.

Suggested config shape:

```yaml
train_sample:
  enabled: false
  frac: 0.5
  stratify_by: ["Universe", "Position_Group"]
  seed: 42
```

This is especially important because universe diversity is more valuable than raw row count.

Implementation targets:

- `fof8-ml/src/fof8_ml/orchestration/data_loader.py`
- `pipelines/conf/`

## Suggested Order Of Work

1. Add universe-level split config and loader support.
2. Extend `PreparedData` to include validation data explicitly.
3. Update classifier/regressor pipelines to score validation and test directly.
4. Replace default CV with grouped CV where CV is still needed.
5. Add optional training-only subsampling for sweeps.
6. Update docs and defaults so grouped-universe evaluation becomes the standard path.

## Practical Default For The Current Dataset

For the current 15-universe setup, the most defensible default is:

- Split by whole universe
- 10 train / 2 validation / 3 test universes
- Tune on validation universes only
- Keep the 3 test universes untouched until the end
- Report final metrics on test universes only after tuning choices are frozen
- Use grouped CV inside the 10 training universes only when comparing close candidates

If runtime becomes an issue:

- run early sweeps on a 50% training subsample
- keep validation and test full
- rerun the best few candidates on the full training universes

## Bottom Line

The main change is conceptual but it needs code support:

- stop validating on rows from universes the model already saw
- start validating on universes the model has never seen
- keep separate test universes that remain unused until final evaluation

The current repository is partway there on metadata and split awareness, but not yet on
evaluation control flow. The biggest gaps are universe-level split assignment, grouped CV,
and actually using held-out universes in model selection and reporting.
