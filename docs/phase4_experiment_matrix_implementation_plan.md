# Phase 4 Experiment Matrix Implementation Plan

This document turns Phase 4 of [`draft_model_finish_line_plan.md`](./draft_model_finish_line_plan.md)
into a concrete repo implementation plan.

It focuses on the missing functionality required to run target/loss experiments in a controlled,
comparable, mostly automated way.

Update after completed Sets A-C and first-pass Set B:

- Sets A and C support the current economic-family champion (`C1_A2`)
- Set B does not justify replacing that champion with a raw overall-target board
- the main remaining research branch is a talent target plus explicit positional-value adjustment,
  which should be treated as a follow-up branch rather than as evidence that raw overall targets
  are sufficient

## Objective

Phase 4 is not blocked on basic metrics anymore. The repo already has:

- configurable target columns and target space
- draft-class-aware ranking metrics
- cross-outcome scorecards
- complete stitched model evaluation
- holdout board CSV generation

What is still missing is the infrastructure to run **experiment matrices** end to end and produce
**comparison-ready outputs** for target/loss decisions.

The implementation target is:

1. define candidate experiment variants in config
2. run classifier + regressor + complete-model evaluation in a controlled way
3. log a stable metric set for every candidate
4. export one normalized comparison table across runs
5. add position-sliced diagnostics so aggregate winners do not hide positional failures
6. preserve manual review artifacts for final model selection

## Existing Repo Capabilities

The following pieces already exist and should be reused rather than replaced.

### Data and target plumbing

- [`data_loader.py`](/workspaces/fof8-scout/fof8-ml/src/fof8_ml/orchestration/data_loader.py)
  already loads:
  - `meta_train`, `meta_val`, `meta_test`
  - `outcomes_train`, `outcomes_val`, `outcomes_test`
  - configurable scorecard outcome columns
- [`economic.yaml`](/workspaces/fof8-scout/pipelines/conf/target/economic.yaml)
  already defines:
  - `classifier_sieve.target_col`
  - `regressor_intensity.target_col`
  - `regressor_intensity.target_space`
  - elite scorecard config

### Metric primitives

- [`metrics.py`](/workspaces/fof8-scout/fof8-ml/src/fof8_ml/evaluation/metrics.py)
  already provides:
  - `ndcg_at_k`
  - `mean_ndcg_by_group`
  - top-k magnitude error and bias metrics
  - calibration slope
- [`evaluator.py`](/workspaces/fof8-scout/fof8-ml/src/fof8_ml/orchestration/evaluator.py)
  already provides:
  - draft-aware regressor metrics
  - cross-outcome scorecard metrics
  - derived elite threshold fitting and application
- [`complete_model.py`](/workspaces/fof8-scout/fof8-ml/src/fof8_ml/evaluation/complete_model.py)
  already provides:
  - stitched complete-model metrics

### MLflow and complete-model evaluation

- [`evaluate_complete_model.py`](/workspaces/fof8-scout/pipelines/evaluate_complete_model.py)
  already evaluates classifier/regressor run pairs
- [`complete_model.py`](/workspaces/fof8-scout/fof8-ml/src/fof8_ml/orchestration/complete_model.py)
  already logs:
  - complete metrics
  - holdout board CSV
  - source run IDs
  - elite config artifact

## Main Gaps Before Phase 4 Can Be Executed Properly

### 1. No experiment-matrix orchestrator

There is no single entrypoint that runs a named candidate through:

1. classifier training or reuse
2. regressor training
3. complete-model evaluation
4. normalized artifact export

Current training and evaluation entrypoints are usable, but they are disconnected.
That makes Phase 4 comparisons manual and error-prone.

### 2. No normalized comparison/export layer

There is no report generator that reads MLflow runs and emits a stable table with rows as
candidates and columns as comparable metrics.

Without this, the repo can produce metrics, but it cannot produce a defensible Phase 4 summary.

### 3. No position-sliced scorecard layer

The current evaluation stack can use `Position_Group` for elite-threshold scoping, but it does not
emit per-position or per-position-group scorecards for:

- complete-model metrics
- cross-outcome metrics
- bust / elite diagnostics

This is the main analytics gap for the decision gate:

> reject candidates that win globally but fail badly for important position groups

### 4. Missing explicit model/config variants for Phase 4

The registry currently exposes CatBoost regression through the key
`catboost_tweedie_regressor`, but Phase 4 requires multiple CatBoost loss variants.

Even if the wrapper can technically accept other CatBoost losses, the repo still needs explicit,
stable config/model keys for:

- CatBoost Tweedie raw
- CatBoost RMSE log1p/raw
- CatBoost MAE raw
- CatBoost Expectile raw

### 5. No automated board-diff artifacts

The holdout board CSV exists, but there is no automated comparison artifact answering:

- which prospects moved most between candidates
- which elite outcomes were missed by each board
- which positions drive the differences
- how much top-32/top-64 overlap exists across candidate boards

### 6. Set D needs derived sieve-target automation

Phase 4 Set D includes classifier target definitions such as:

- `Career_Merit_Cap_Share > 0`
- `Career_Games_Played >= 16`
- `Career_Games_Played >= 48`
- top position-relative economic percentile

Only the first definition is already represented cleanly in config. The others need target-building
support or generic derived-binary-target support.

## Implementation Strategy

Implement Phase 4 in four layers:

1. candidate definition layer
2. execution/orchestration layer
3. evaluation/reporting layer
4. review artifact layer

This keeps the system composable and avoids hardcoding one-off experiment scripts.

## Layer 1: Candidate Definition Layer

### Goal

Represent each experiment candidate as declarative config rather than shell-command folklore.

### Required additions

#### A. Add experiment matrix config group

Create a new config area:

- `pipelines/conf/matrix/`

Suggested files:

- `pipelines/conf/matrix/set_a_economic.yaml`
- `pipelines/conf/matrix/set_b_overall.yaml`
- `pipelines/conf/matrix/set_c_ablation.yaml`
- `pipelines/conf/matrix/set_d_classifier_sieve.yaml`
- `pipelines/conf/matrix/set_e_talent_position_adjusted.yaml`
- `pipelines/conf/matrix/common.yaml`

Suggested config contract:

```yaml
matrix_name: "phase4_set_a_economic"
shared:
  split: grouped_universe
  seed: 42
  use_gpu: false
  optuna_trials: 20
  feature_profile: default
  classifier_source: fixed_run | train_with_candidate | named_candidate
  fixed_classifier_run_id: null

candidates:
  - candidate_id: "A1"
    label: "positive_merit_tweedie_raw"
    classifier:
      target_profile: economic
    regressor:
      model: catboost_regressor_tweedie
      target_col: Positive_Career_Merit_Cap_Share
      target_space: raw
      loss_function: Tweedie
  - candidate_id: "A2"
    label: "positive_merit_rmse_log"
    classifier:
      target_profile: economic
    regressor:
      model: catboost_regressor_rmse
      target_col: Positive_Career_Merit_Cap_Share
      target_space: log
      loss_function: RMSE
```

Design rule:

- use scalar config values where possible
- avoid complex list/dict categorical values in sweep search spaces
- keep candidate definitions directly serializable into MLflow params/tags

Future branch note:

- Set E should support post-regressor adjustment metadata or explicit candidate descriptors for the
  positional-value layer so the comparison table can distinguish raw talent boards from
  talent-plus-adjustment boards

#### B. Add target-profile overlays

Add or extend target config overlays to support Phase 4 Set D classifier definitions.

Suggested files:

- `pipelines/conf/target/economic_success.yaml`
- `pipelines/conf/target/games_played_16.yaml`
- `pipelines/conf/target/games_played_48.yaml`
- `pipelines/conf/target/position_relative_econ_percentile.yaml`

If target-profile sprawl becomes too large, use a single target config with named profiles and a
resolver in orchestration.

### Acceptance criteria

- every Phase 4 candidate can be described with config only
- candidate config fully determines target, loss, target space, ablation profile, and source
  classifier behavior
- candidate configs are machine-readable and stable enough for reporting

## Layer 2: Execution/Orchestration Layer

### Goal

Run one Phase 4 candidate end to end with a single command.

### Required additions

#### A. Add a Phase 4 runner entrypoint

Create:

- `pipelines/run_experiment_matrix.py`

Responsibilities:

1. load matrix config
2. materialize shared settings
3. loop over candidates
4. train or resolve classifier run
5. train regressor run
6. invoke complete-model evaluation
7. write a per-candidate manifest artifact
8. optionally write a matrix-level manifest

The runner should not duplicate training logic. It should call the existing entrypoints/modules.

Preferred architecture:

- thin Hydra pipeline script in `pipelines/`
- orchestration logic in `fof8-ml/src/fof8_ml/orchestration/experiment_matrix.py`

#### B. Add reusable candidate execution module

Create:

- `fof8-ml/src/fof8_ml/orchestration/experiment_matrix.py`

Suggested public API:

- `resolve_phase4_candidates(cfg) -> list[Phase4Candidate]`
- `run_phase4_candidate(candidate, common_cfg, logger, exp_root) -> Phase4CandidateResult`
- `run_phase4_matrix(cfg, logger, exp_root) -> list[Phase4CandidateResult]`

Suggested dataclasses:

- `Phase4Candidate`
- `Phase4CandidateRuns`
- `Phase4CandidateResult`
- `Phase4MatrixSummary`

#### C. Add per-candidate manifests

Each candidate should produce a normalized JSON artifact containing:

```json
{
  "candidate_id": "A1",
  "label": "positive_merit_tweedie_raw",
  "classifier_run_id": "...",
  "regressor_run_id": "...",
  "complete_run_id": "...",
  "classifier_target_col": "Economic_Success",
  "regressor_target_col": "Positive_Career_Merit_Cap_Share",
  "target_space": "raw",
  "loss_function": "Tweedie",
  "ablation_signature": "...",
  "elite_config": {
    "source_column": "Career_Merit_Cap_Share",
    "quantile": 0.95,
    "scope": "position_group",
    "fallback_scope": "global"
  }
}
```

These manifests become the source of truth for later comparison and reruns.

#### D. Classifier reuse policy

Phase 4 explicitly says classifier behavior should stay fixed while isolating regressor target/loss.
The runner should support:

- `fixed_run`: use one previously trained classifier run for all candidates in a set
- `train_once_per_matrix`: train one classifier under shared settings and reuse it
- `train_per_candidate`: train classifier per candidate when the candidate changes sieve definition

This distinction matters because Set A/B/C and Set D have different needs.

After first-pass Set B, the next distinct need is:

- `postprocess_per_candidate`: reuse a talent regressor run but apply a train-only learned
  positional-value transformation before complete-model evaluation

That future Set E branch should remain separate from the raw Set B interpretation because it tests
a different hypothesis.

### Acceptance criteria

- one command can execute a full matrix or a selected candidate subset
- source run IDs are automatically captured
- no manual copy/paste of run IDs is required for normal workflow

## Layer 3: Evaluation And Reporting Layer

### Goal

Export a stable, comparable scorecard across candidates.

Post-Set-B interpretation rule:

- reports should expose both self-target stitched metrics and cross-outcome metrics
- cross-family decisions involving Set B or Set E should be based primarily on the cross-outcome
  table, not on the self-target stitched score alone

### Required additions

#### A. Add matrix comparison exporter

Create:

- `pipelines/export_matrix_report.py`
- `fof8-ml/src/fof8_ml/reporting/matrix_report.py`

Responsibilities:

1. read matrix manifest or discover runs from MLflow tags
2. resolve candidate metadata
3. load complete-model metrics
4. flatten scorecards into one row per candidate
5. write:
   - CSV
   - parquet
   - markdown summary

Suggested output paths:

- `outputs/phase4/phase4_candidate_summary.csv`
- `outputs/phase4/phase4_candidate_summary.parquet`
- `outputs/phase4/phase4_candidate_summary.md`

#### B. Define the canonical comparison table schema

Minimum columns:

- `candidate_id`
- `label`
- `classifier_target_col`
- `regressor_target_col`
- `target_space`
- `loss_function`
- `feature_ablation_signature`
- `classifier_run_id`
- `regressor_run_id`
- `complete_run_id`
- `complete_mean_ndcg_at_64`
- `complete_top64_weighted_mae_normalized`
- `complete_top64_bias`
- `complete_top64_calibration_slope`
- `complete_top64_actual_value`
- `complete_bust_rate_at_32`
- `complete_elite_precision_at_32`
- `complete_elite_recall_at_64`
- `complete_draft_value_score`
- `complete_econ_mean_ndcg_at_64`
- `complete_talent_mean_ndcg_at_64`
- `complete_longevity_mean_ndcg_at_64`
- `elite_source_column`
- `elite_quantile`
- `elite_scope`
- `elite_fallback_scope`

Also include availability flags so missing outcome families remain explicit.

#### C. Add optional portfolio score computation

Implement in reporting, not core training.

Rationale:

- the portfolio score is a decision aid
- it should not redefine raw model metrics
- it may change as decision policy changes

Suggested module:

- `fof8-ml/src/fof8_ml/reporting/portfolio_score.py`

Suggested function:

- `compute_portfolio_score(row, weights_cfg) -> float`

The report exporter should be able to append:

- `portfolio_score`
- `portfolio_score_version`
- `portfolio_weight_config`

#### D. Add MLflow tagging conventions for discovery

Ensure candidate runs are tagged with at least:

- `phase=4`
- `matrix_name=<...>`
- `candidate_id=<...>`
- `candidate_label=<...>`
- `experiment_set=A|B|C|D`
- `evaluation_type=complete_model`

This makes report generation robust even if manifest files move.

### Acceptance criteria

- one exporter command can produce a Phase 4 summary table from run metadata
- the output schema is stable across sets A/B/C/D
- report rows remain comparable even when some optional labels are unavailable

## Layer 4: Position-Sliced Diagnostics

### Goal

Add automated per-position diagnostics so aggregate winners can be rejected when they fail on
important position groups.

### Required additions

#### A. Add generic sliced-metric utilities

Extend or add:

- `fof8-ml/src/fof8_ml/evaluation/sliced_metrics.py`

Suggested functions:

- `compute_metrics_by_slice(metric_fn, y_true, y_score, slice_values, ...)`
- `compute_complete_metrics_by_slice(...)`
- `compute_cross_outcome_metrics_by_slice(...)`

Design rule:

- keep the existing aggregate metric functions unchanged
- build slicing as a wrapper layer around them

#### B. Emit position-group summaries from complete-model evaluation

Extend:

- [`complete_model.py`](/workspaces/fof8-scout/fof8-ml/src/fof8_ml/orchestration/complete_model.py)

New artifact suggestion:

- `outputs/complete_model_position_group_metrics.csv`

Minimum columns:

- `slice_column`
- `slice_value`
- `n_players`
- `n_draft_classes`
- `complete_mean_ndcg_at_64`
- `complete_top64_actual_value`
- `complete_top64_weighted_mae_normalized`
- `complete_top64_bias`
- `complete_bust_rate_at_32`
- `complete_elite_precision_at_32`
- `complete_elite_recall_at_64`

Default slice dimensions:

- `Position_Group`

Optional later extension:

- exact `Position`
- `Universe`
- `Universe x Position_Group`

#### C. Add position-specific comparison export

The Phase 4 summary exporter should write a second table:

- `outputs/phase4/phase4_position_group_summary.csv`

Schema:

- `candidate_id`
- `slice_column`
- `slice_value`
- sliced metric columns

This becomes the basis for Phase 5 failure notes.

#### D. Add failure-flag heuristics in reporting

The report layer should compute flags such as:

- `fails_position_group_bias_gate`
- `fails_position_group_bust_gate`
- `fails_position_group_ndcg_gate`

Example rule:

- fail if any premium position group has `complete_bust_rate_at_32` worse than baseline by more
  than configured tolerance

These should be decision-support flags, not hard training constraints.

### Acceptance criteria

- every complete-model evaluation can emit aggregate and position-group scorecards
- Phase 4 report can show both global and position-group results per candidate
- Phase 5 can refer to machine-generated position-specific failure notes

## Layer 5: Model And Target Support Needed For Phase 4

### Goal

Make every planned candidate executable without ad hoc overrides.

### Required additions

#### A. Register explicit CatBoost regressor model keys

Extend:

- [`registry.py`](/workspaces/fof8-scout/fof8-ml/src/fof8_ml/models/registry.py)

Add model keys such as:

- `catboost_regressor_tweedie`
- `catboost_regressor_rmse`
- `catboost_regressor_mae`
- `catboost_regressor_expectile`

Add matching config files:

- `pipelines/conf/model/catboost_regressor_tweedie.yaml`
- `pipelines/conf/model/catboost_regressor_rmse.yaml`
- `pipelines/conf/model/catboost_regressor_mae.yaml`
- `pipelines/conf/model/catboost_regressor_expectile.yaml`

The wrapper should continue to compose CatBoost-specific loss syntax where needed.

#### B. Support Expectile parameters

If CatBoost Expectile requires an alpha-style parameter encoded inside loss syntax, add explicit
wrapper support in:

- `fof8-ml/src/fof8_ml/models/catboost_wrapper.py`

That support should be generic enough to avoid another one-off branch for future asymmetric losses.

#### C. Add derived classifier-target support for Set D

Set D should not be implemented through manual column editing.

Preferred approach:

- add derived binary outcome builders in `fof8-core`
- expose them in processed features
- select them through target config

Likely target additions:

- `Games_Played_At_Least_16`
- `Games_Played_At_Least_48`
- `Position_Relative_Economic_Elite`

Likely files:

- `fof8-core/src/fof8_core/targets/`
- `fof8-core/src/fof8_core/targets/draft_outcomes.py`
- `fof8-ml/src/fof8_ml/orchestration/data_loader.py`
- `pipelines/process_features.py`

### Acceptance criteria

- all Set A/B/C/D candidates are runnable without custom shell overrides
- target/loss selection is represented explicitly in config and MLflow metadata

## Layer 6: Review Artifacts And Board Diffs

### Goal

Automate the manual-review artifacts needed for final champion selection.

### Required additions

#### A. Add board comparison exporter

Create:

- `pipelines/export_phase4_board_diffs.py`
- `fof8-ml/src/fof8_ml/reporting/board_diffs.py`

Inputs:

- two or more complete-model board CSV artifacts

Outputs:

- top-32 overlap summary
- top-64 overlap summary
- largest rank deltas by candidate pair
- elite-hit and bust-miss deltas
- position-group composition differences in top-32/top-64

Suggested artifacts:

- `outputs/phase4/board_overlap_summary.csv`
- `outputs/phase4/board_rank_deltas.csv`
- `outputs/phase4/board_position_mix.csv`

#### B. Add candidate notes template generation

Create a markdown generator that pre-populates one section per candidate with:

- global metrics
- position-group summary
- biggest board differences vs baseline
- placeholders for manual notes

Suggested output:

- `outputs/phase4/phase4_review_template.md`

This does not replace analyst judgment. It reduces clerical work and makes notes consistent.

### Acceptance criteria

- manual board review starts from generated artifacts, not raw MLflow clicking
- review notes can be tied back to a candidate ID and source run IDs

## Recommended Build Order

Implement in this order.

### Step 1. Add model/config support for all candidate losses

Files:

- `fof8-ml/src/fof8_ml/models/registry.py`
- `fof8-ml/src/fof8_ml/models/catboost_wrapper.py`
- `pipelines/conf/model/*.yaml`

Reason:

- the matrix runner is not useful until candidate models are first-class config options

### Step 2. Add Set D classifier-target support

Files:

- `fof8-core/src/fof8_core/targets/`
- `pipelines/conf/target/*.yaml`
- `pipelines/process_features.py`

Reason:

- Set D otherwise remains informal and incomplete

### Step 3. Add the Phase 4 matrix runner

Files:

- `pipelines/run_experiment_matrix.py`
- `fof8-ml/src/fof8_ml/orchestration/experiment_matrix.py`

Reason:

- this is the core automation layer

### Step 4. Add position-sliced evaluation artifacts

Files:

- `fof8-ml/src/fof8_ml/evaluation/sliced_metrics.py`
- `fof8-ml/src/fof8_ml/orchestration/complete_model.py`

Reason:

- aggregate comparisons alone are not sufficient for decision gates

### Step 5. Add report exporters

Files:

- `pipelines/export_matrix_report.py`
- `fof8-ml/src/fof8_ml/reporting/matrix_report.py`
- `fof8-ml/src/fof8_ml/reporting/portfolio_score.py`

Reason:

- this produces the actual Phase 4 summary tables

### Step 6. Add board-diff exporters

Files:

- `pipelines/export_phase4_board_diffs.py`
- `fof8-ml/src/fof8_ml/reporting/board_diffs.py`

Reason:

- these artifacts support the final human review loop

## Testing Plan

### Unit tests

Add tests for:

- candidate config resolution
- classifier reuse policy resolution
- report table flattening
- portfolio score computation
- sliced metrics by `Position_Group`
- board-diff overlap and rank delta calculations

Likely locations:

- `fof8-ml/tests/test_phase4_matrix.py`
- `fof8-ml/tests/test_phase4_report.py`
- `fof8-ml/tests/test_sliced_metrics.py`
- `fof8-ml/tests/test_board_diffs.py`

### Integration tests

Add a smoke path that:

1. runs a tiny matrix with 2 candidates
2. produces candidate manifests
3. runs complete-model evaluation
4. exports a comparison table

This can use monkeypatched MLflow interactions where necessary.

### Regression checks

Preserve or extend tests around:

- missing optional outcome labels
- elite-threshold fallback by `Position_Group`
- draft-group-aware NDCG behavior
- target-space safety rules for Tweedie/raw

## Suggested Commands After Implementation

Examples only; exact CLI shape can change.

Run one matrix:

```bash
uv run python pipelines/run_experiment_matrix.py matrix=set_a_economic
```

Run selected candidates only:

```bash
uv run python pipelines/run_experiment_matrix.py matrix=set_a_economic candidate_ids=[A1,A2]
```

Export comparison report:

```bash
uv run python pipelines/export_matrix_report.py matrix_name=phase4_set_a_economic
```

Export board diffs:

```bash
uv run python pipelines/export_phase4_board_diffs.py matrix_name=phase4_set_a_economic
```

## Definition Of Done For This Document

Phase 4 is operational when the repo can do all of the following without manual run-ID juggling:

1. execute a named experiment matrix from config
2. reuse or retrain classifier runs according to matrix policy
3. train regressor candidates with explicit target/loss/target-space contracts
4. evaluate stitched complete-model performance on the held-out split
5. export one normalized candidate summary table
6. export position-group diagnostics
7. export board-diff review artifacts
8. preserve enough metadata to reproduce every candidate exactly

## Recommended First Deliverable

The first implementation milestone should be:

- explicit CatBoost regressor variants
- `run_phase4_matrix.py`
- candidate manifest artifacts
- `export_phase4_report.py`

That subset is enough to make Phase 4 comparisons real. Position-sliced diagnostics and board
comparison artifacts should follow immediately after, but they do not need to block the first
controlled matrix run.
