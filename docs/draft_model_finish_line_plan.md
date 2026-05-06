# Draft Model Finish-Line Implementation Plan

This plan turns the target-strategy work into concrete repo changes and experiments. It is
intended to answer the open modeling questions and leave the project with a defensible
draft-board model.

Related strategy doc: [`draft_value_target_strategy.md`](./draft_value_target_strategy.md)

## Finish-Line Definition

The project is at the finish line when:

- The pipeline can train classifier and regressor variants against configurable targets.
- Regressor evaluation logs both ranking quality and magnitude calibration.
- Complete classifier + regressor evaluation works on full held-out draft classes.
- Target/loss experiments have been run with comparable settings.
- A champion target, loss, and model family have been selected by documented decision
  criteria.
- The final draft-board score is usable for ranking and trade-value reasoning.

## Guiding Decisions

Use these as the default direction unless experiments overturn them:

```text
Primary target candidate:
  clipped_positive_Career_Merit_Cap_Share

Primary regressor sweep objective:
  mean_ndcg_at_64 - 0.25 * top64_weighted_mae_normalized

Primary loss candidates:
  1. raw clipped economic target + CatBoost Tweedie
  2. log1p clipped economic target + CatBoost RMSE

Final champion selection:
  complete stitched classifier + regressor metrics on full held-out draft classes

Secondary output:
  predicted Peak_Overall or position-relative talent percentile
```

## Phase 1: Target Contract Refactor

Goal: make targets configurable enough to compare DPO, economic value, and overall-based
targets without rewriting pipeline code.

Target-family convention:

- `Peak_Overall` is a talent target.
- `Career_Merit_Cap_Share` is an economic target.
- `DPO` is a composite target (talent x economic) and should stay a baseline/comparison target.

### Repo Changes

1. Add target columns to the processed feature store.

   Current processed data already contains:

   - `Peak_Overall`
   - `Career_Games_Played`
   - `Cleared_Sieve`
   - `DPO`
   - `Career_Merit_Cap_Share`

   Add derived target columns in the feature processing or data-loading layer:

   - `Positive_Career_Merit_Cap_Share = max(Career_Merit_Cap_Share, 0)`
   - `Positive_DPO = max(DPO, 0)`
   - `Top3_Mean_Current_Overall` if it differs from existing `Peak_Overall`
   - optional `Economic_Success = Career_Merit_Cap_Share > 0`

   Likely files:

   - `fof8-ml/src/fof8_ml/data/economic_dataset.py`
   - `fof8-ml/src/fof8_ml/orchestration/data_loader.py`
   - `pipelines/process_features.py`

2. Update target config.

   Extend `pipelines/conf/target/economic.yaml` from a fixed DPO setup to a named target
   profile:

   ```yaml
   classifier_sieve:
     target_col: "Economic_Success"
     merit_threshold: 0
     min_positive_recall: 0.95

   regressor_intensity:
     target_col: "Positive_Career_Merit_Cap_Share"
     target_space: "raw"
     target_family: "economic"
   ```

3. Make target space explicit.

   Current target-space detection is model/loss-specific in
   `fof8_ml.orchestration.regressor._regressor_target_space`. Replace or extend it so
   config can control target transformation:

   ```yaml
   regressor_intensity:
     target_space: "raw"  # raw | log
   ```

   Keep the CatBoost Tweedie safety rule: Tweedie must use non-negative raw targets.

### Acceptance Criteria

- Training still works with existing DPO target.
- Training can switch to `Positive_Career_Merit_Cap_Share` via config only.
- Logged config clearly identifies target column and target space.
- Feature schema still excludes all target columns from model features.

## Phase 2: Draft-Aware And Cross-Outcome Metrics

Goal: evaluate regressors by draft-board usefulness, not only RMSE/MAE, and avoid
choosing a target by only measuring how well a model predicts that same target.

You have completed Phase 1 and are starting here. The most important Phase 2 design
choice is that `NDCG@K` should be computed per draft class:

```text
for each draft year:
  rank that year's prospects by model score
  compute NDCG@K within that year
average NDCG@K across years
```

Do not use global historical NDCG as the primary metric. The real decision is made within
one live draft class, and global NDCG can be dominated by unusually strong or weak years.

### Repo Changes

1. Add metric utilities.

   Create or extend:

   - `fof8-ml/src/fof8_ml/evaluation/metrics.py`

   Add:

   - `ndcg_at_k`
   - `mean_ndcg_by_group`
   - `topk_weighted_mae`
   - `topk_weighted_mae_normalized`
   - `topk_bias`
   - `calibration_slope`
   - optional `spearman_by_group`

   `mean_ndcg_by_group` should take `group=draft_year` and apply the top-K cutoff inside
   each group, not after globally ranking all players.

2. Pass draft year metadata into regressor metric calculation.

   `PreparedData` already carries `meta_train`; use `Year` as the draft-class group.

   Likely files:

   - `fof8-ml/src/fof8_ml/orchestration/regressor.py`
   - `fof8-ml/src/fof8_ml/orchestration/evaluator.py`
   - `fof8-ml/src/fof8_ml/orchestration/pipeline_types.py`

3. Add same-target conditional regressor metrics.

   For the standalone regressor, compute metrics on the success/positive subset:

   ```text
   regressor_mean_ndcg_at_32
   regressor_mean_ndcg_at_64
   regressor_top64_weighted_mae
   regressor_top64_weighted_mae_normalized
   regressor_top64_bias
   regressor_top64_calibration_slope
   regressor_rmse_positive
   regressor_mae_positive
   ```

4. Add cross-outcome evaluation metrics.

   Target selection should not be circular. A model trained on economic value should be
   checked against talent, longevity, and elite outcomes. A model trained on overall should
   be checked against economic value and draft utility.

   Add a generic evaluation path that accepts:

   ```text
   y_score = model predictions
   outcome_columns = independent outcome labels
   group = draft year
   ```

   Evaluate the same ranked board against multiple outcome families:

   ```text
   Economic:
     Positive_Career_Merit_Cap_Share
     Career_Merit_Cap_Share
     Positive_DPO

   Talent:
     Peak_Overall
     Top3_Mean_Current_Overall if available

   Longevity:
     Career_Games_Played

   Elite outcomes:
     Award_Count if available
     Hall_of_Fame_Flag or HOF points if available
   ```

   The comparison should produce metrics like:

   ```text
   cross_econ_mean_ndcg_at_64
   cross_talent_mean_ndcg_at_64
   cross_longevity_mean_ndcg_at_64
   cross_elite_precision_at_64
   cross_econ_top64_actual_value
   cross_bust_rate_at_32
   ```

   If some outcome columns are not available yet, build the metric API so they can be added
   without changing model-training code.

5. Add composite metric.

   ```text
   regressor_draft_value_score =
     regressor_mean_ndcg_at_64 - 0.25 * regressor_top64_weighted_mae_normalized
   ```

   Set this as the default regressor optimization metric once stable.

   Treat this as a within-target sweep objective. For target selection across economic vs
   talent vs hybrid targets, use the cross-outcome scorecard rather than this scalar alone.

6. Add an optional portfolio score for target-selection summaries.

   A target-selection report can include a transparent weighted score:

   ```text
   portfolio_score =
     0.45 * cross_econ_mean_ndcg_at_64
   + 0.25 * cross_talent_mean_ndcg_at_64
   + 0.15 * cross_longevity_mean_ndcg_at_64
   + 0.15 * cross_elite_precision_at_64
   - 0.20 * cross_bust_rate_at_32
   - 0.15 * top64_value_miscalibration
   ```

   These weights are not objective truth. They are a decision aid that makes tradeoffs
   explicit. The final decision should still inspect the full scorecard and draft-board
   artifacts.

### Acceptance Criteria

- Metrics are logged to MLflow.
- DVC metric output can write the selected composite score.
- NDCG@K metrics apply K within each draft year and then average across years.
- Cross-outcome metrics can evaluate one model's board against economic, talent,
  longevity, and elite outcome labels when those labels are present.
- Regression tests cover edge cases:
  - all-zero relevance class
  - fewer than K players in a group
  - negative target values clipped for ranking relevance
  - missing optional cross-outcome labels are skipped or reported clearly

## Phase 3: Complete Stitched Model Evaluation

Goal: evaluate the classifier + regressor as the draft board will actually use them.

### Repo Changes

1. Add a complete-model evaluator module or script.

   Candidate file:

   - `pipelines/evaluate_complete_model.py`

   Responsibilities:

   - Accept classifier run ID and regressor run ID.
   - Load `feature_schema.json` for each run.
   - Load `classifier_model/` and `regressor_model/`.
   - Apply each schema separately to the full evaluation feature frame.
   - Compute:

     ```text
     complete_pred = P(success | X) * max(regressor_pred, 0)
     ```

2. Add reusable loader helpers.

   Candidate file:

   - `fof8-ml/src/fof8_ml/evaluation/complete_model.py`

   Functions:

   - `load_feature_schema(client, run_id)`
   - `load_catboost_complete_model(classifier_run_id, regressor_run_id)`
   - `predict_complete_model(X_full, classifier_bundle, regressor_bundle)`
   - `evaluate_complete_model(y_true, y_pred, draft_year)`

3. Log complete-model metrics.

   ```text
   complete_mean_ndcg_at_32
   complete_mean_ndcg_at_64
   complete_top32_actual_value
   complete_top64_actual_value
   complete_top32_weighted_mae_normalized
   complete_top64_weighted_mae_normalized
   complete_top64_bias
   complete_top64_calibration_slope
   complete_bust_rate_at_32
   complete_precision_at_32_positive_value
   ```

4. Add evaluation artifacts.

   Log a CSV with one row per held-out prospect:

   ```text
   Player_ID
   Year
   Position_Group
   classifier_probability
   regressor_prediction
   complete_prediction
   actual_target
   rank_within_year
   ```

### Acceptance Criteria

- Complete evaluator can compare any compatible classifier/regressor run pair.
- Evaluator fails clearly when a required schema/model artifact is missing.
- Complete metrics can be used for final champion selection.
- The output CSV is manually inspectable as a draft board.

## Phase 4: Loss And Target Experiment Matrix

Goal: answer the core modeling questions empirically.

### Controlled Setup

Keep constant:

- train/test split
- feature ablation profile
- model family where applicable
- number of Optuna trials per experiment
- random seed policy
- classifier model when isolating regressor target/loss

### Experiment Set A: Economic Targets

Run:

```text
A1: Positive_Career_Merit_Cap_Share + CatBoost Tweedie raw
A2: Positive_Career_Merit_Cap_Share + CatBoost RMSE log1p
A3: Positive_Career_Merit_Cap_Share + CatBoost MAE raw
A4: Positive_Career_Merit_Cap_Share + CatBoost Expectile alpha=0.7 raw
A5: Positive_DPO + CatBoost Tweedie raw
```

Primary comparison:

- complete stitched `draft_value_score`
- complete top64 weighted MAE normalized
- complete bias/calibration
- cross-outcome scorecard: economic, talent, longevity, and elite labels

### Experiment Set B: Overall Targets

Run:

```text
B1: Peak_Overall + CatBoost RMSE
B2: Peak_Overall + CatBoost MAE
B3: Top3_Mean_Current_Overall + CatBoost RMSE
```

Compare whether these produce better draft boards under:

- actual economic value
- actual peak overall
- games played
- awards/HOF labels when available
- bust avoidance
- manual inspection

The key question is not "does the overall-trained model predict overall?" It is whether
the overall-trained board holds up against economic and draft-utility outcomes better than
the economic-trained board holds up against talent outcomes.

### Experiment Set C: Feature/Ablation Sensitivity

Run the leading target/loss under:

```text
C1: current default ablation
C2: keep College for CatBoost
C3: keep scout features
C4: keep delta features
C5: remove Position and keep only Position_Group
```

Goal: ensure the champion is not only winning because of one fragile feature-policy choice.

### Experiment Set D: Classifier Threshold/Sieve Definition

Run:

```text
D1: Economic_Success = Career_Merit_Cap_Share > 0
D2: Career_Games_Played >= 16
D3: Career_Games_Played >= 48
D4: top position-relative economic percentile
```

Evaluate complete model quality, not just classifier PR-AUC.

### Acceptance Criteria

- Every experiment logs the same metric set.
- Every experiment has a run name/tag identifying target, loss, target space, and feature
  ablation signature.
- Results are summarized in a table or MLflow comparison report with rows as trained
  target/loss variants and columns as cross-outcome evaluation families.

## Phase 5: Champion Decision Process

Goal: choose the final target/loss/model with documented rationale.

### Required Scorecard

For each candidate, record:

```text
target
loss_function
target_space
feature_ablation_signature
complete_mean_ndcg_at_64
complete_top64_weighted_mae_normalized
complete_top64_bias
complete_top64_calibration_slope
complete_top64_actual_value
complete_bust_rate_at_32
cross_econ_mean_ndcg_at_64
cross_talent_mean_ndcg_at_64
cross_longevity_mean_ndcg_at_64
cross_elite_precision_at_64
portfolio_score if used
position-specific failure notes
manual draft-board notes
```

### Decision Gates

Reject a candidate if:

- top64 bias is materially positive and calibration cannot be corrected
- it wins only on the metric it was trained to predict and fails reasonable cross-outcome
  checks
- it wins globally but fails badly for important position groups
- it relies on target leakage or unavailable inference features
- it produces a draft board that is not manually credible

Prefer a candidate if:

- it ranks high-value players well within draft classes
- it is robust across economic, talent, longevity, and elite outcome families
- it has useful magnitude calibration in top-k decision regions
- it generalizes across held-out years and positions
- it supports trade decisions with interpretable units

## Phase 6: Productionization

Goal: make the selected model usable.

### Repo Changes

1. Add a combined draft-board prediction entrypoint.

   Candidate:

   - `pipelines/build_draft_board.py`

   Inputs:

   - classifier run ID or registered model name
   - regressor run ID or registered model name
   - draft year / feature file

   Outputs:

   - CSV/parquet draft board
   - optional HTML/report artifact

2. Register champion models.

   The existing sweep manager already registers role models. Extend or document champion
   pairing:

   ```text
   champion_classifier_run_id
   champion_regressor_run_id
   champion_target
   champion_score
   ```

3. Add calibration layer if needed.

   If the best model ranks well but over/understates value, add a held-out calibration step
   for the regressor or complete score:

   - isotonic regression for monotonic calibration
   - linear calibration for simple slope/intercept correction

4. Add final documentation.

   Update:

   - `fof8-ml/README.md`
   - `docs/draft_value_target_strategy.md`
   - this plan with final chosen decisions

### Acceptance Criteria

- A user can produce a draft board from trained model runs.
- The board includes prediction, rank, probability, conditional value, and confidence/context
  fields.
- The model pair and schemas are reproducible from MLflow artifacts.

## Suggested Implementation Order

1. Implement metric utilities and tests.
2. Add draft-aware regressor metrics to current DPO pipeline.
3. Add complete stitched evaluator using existing DPO runs.
4. Add positive economic target columns/config.
5. Run small smoke experiments for DPO vs positive economic target.
6. Add log-RMSE loss family support if needed.
7. Run full target/loss matrix.
8. Run feature ablation sensitivity matrix.
9. Choose champion with scorecard.
10. Build production draft-board entrypoint.

This order gives useful feedback early. The existing DPO pipeline becomes the baseline,
then each new target/loss/feature decision is evaluated against it.

## Concrete Initial Tickets

### Ticket 1: Add Draft Metrics

- Add metric functions to `fof8_ml.evaluation.metrics`.
- Add unit tests under `fof8-ml/tests/`.
- Include grouped NDCG, top-k weighted MAE, bias, calibration slope.

### Ticket 2: Log Regressor Draft Metrics

- Update `compute_regressor_oof_metrics`.
- Pass draft year from `meta_train`.
- Log composite `regressor_draft_value_score`.
- Set `pipelines/conf/regressor_pipeline.yaml` optimization metric to the composite.

### Ticket 3: Add Complete Model Evaluator

- Add reusable model/schema loading helpers.
- Add CLI script for classifier run ID + regressor run ID.
- Log complete metrics and output draft-board CSV.

### Ticket 4: Add Positive Economic Target

- Add `Positive_Career_Merit_Cap_Share` and `Economic_Success`.
- Update target config to select them.
- Ensure target columns are excluded from features and schema.

### Ticket 5: Add Loss/Target Experiment Configs

- Add model configs for Tweedie raw and log-RMSE CatBoost.
- Add experiment configs under `pipelines/conf/experiment/`.
- Add search spaces under `pipelines/conf/hparams_search/`.

### Ticket 6: Run And Summarize Experiment Matrix

- Run controlled sweeps.
- Export MLflow comparison.
- Write final champion notes into this document or a separate experiment report.
