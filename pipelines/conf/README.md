# Pipeline Config Layout

This directory contains Hydra config groups used by training and inference entrypoints.

## Root Pipeline Configs

- `classifier_pipeline.yaml`: classifier training defaults and Hydra runtime settings.
- `regressor_pipeline.yaml`: regressor training defaults and Hydra runtime settings.
- `complete_model_pipeline.yaml`: complete-model evaluation defaults and runtime inputs.
- `experiment_matrix_pipeline.yaml`: first-pass target/loss matrix runner defaults.
- `matrix_report_pipeline.yaml`: Matrix comparison report export defaults.
- `matrix_diagnostics_pipeline.yaml`: Matrix second-pass diagnostics export defaults.

Both root pipeline configs expose `runtime.refit_final_model`.

- `false` (default): validation-holdout runs stop after validation metrics and do not refit on
  `train + validation` or score the held-out test universes. This is the preferred setting for
  hyperparameter sweeps and most iterative tuning work.
- `true`: after tuning on validation, refit a final model on `train + validation` and score the
  held-out test universes. Use this for finalist or final-report runs once the configuration is
  stable.

## Config Groups

- `ablation/`: shared feature-ablation definitions and defaults.
- `data/`: dataset source and column wiring.
- `target/`: target builder selection.
- `split/`: train/validation split strategy.
- `cv/`: cross-validation setup.
- `model/`: model family + wrapper parameters.
- `hparams_search/`: Optuna/Hydra search spaces.
- `experiment/`: reusable sweep presets and run metadata.
- `matrix/`: declarative candidate matrices for Matrix target/loss comparisons.


## Data And Split Configuration

Raw data can be loaded from one or more universe folders under `data.raw_path`. Prefer
`data.league_names` for explicit pooled training runs; `data.league_name` remains as a
legacy single-universe fallback. `data.league_glob` may be used by transform jobs to
discover matching folders.

Use `data.year_start_offset` and `data.year_count` to materialize a relative window per
universe. For example, `data.year_start_offset=1 data.year_count=30` starts from each
universe's second simulation year and includes up to 30 years.

The transform stage writes both the pooled `data.features_path` parquet and per-universe
parquet files at `fof8-ml/data/processed/universes/<Universe>/features.parquet`.

Split configs live under `split/`:

- `chronological`: applies the right-censor buffer and holds out the latest eligible draft classes within each universe before pooling train/test rows.
- `random`: applies the same eligibility filter, then randomly splits either rows or whole `(Universe, Year)` draft classes with `split.seed`.
- `grouped_universe`: applies the eligibility filter, then assigns whole universes to train/validation/test splits.

For `split.unit=draft_class`, the grouping key is the full `(Universe, Year)` pair rather than
numeric `Year` alone. That is the default for random splits because it preserves whole draft
boards during pooled-universe training. Row-level random split remains available for diagnostics,
while chronological split is still the preferred final backtest/reporting holdout.

`grouped_universe` is the default training configuration for unseen-universe generalization.
It exposes an explicit validation universe set for model selection and keeps test universes
untouched until final reporting.

In practice, that means:

- default tuning runs should use `runtime.refit_final_model=false`
- final-report runs can use `runtime.refit_final_model=true`

Example overrides:

```bash
uv run python pipelines/train_classifier.py split=random
uv run python pipelines/train_regressor.py data.league_names=[DRAFT003,DRAFT004,DRAFT005]
uv run python pipelines/train_classifier.py runtime.refit_final_model=true
```

## Shared Ablation Source Of Truth

Feature ablation groups/toggles are centralized in:

- `ablation/default.yaml`

Both classifier and regressor pipelines import this via:

- `defaults: - ablation: default`

This keeps `ablation.groups`, `ablation.toggle_to_group`, and default toggle values in one
place so sweeps and fixed runs stay aligned across pipelines.

## Editing Guidance

When adding a new ablation toggle:

1. Add the group patterns in `ablation/default.yaml` under `ablation.groups`.
2. Add the toggle in `ablation.toggles`.
3. Map it in `ablation.toggle_to_group`.
4. Optionally expose it in `hparams_search/*.yaml` as a boolean categorical.
