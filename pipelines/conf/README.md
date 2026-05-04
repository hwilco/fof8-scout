# Pipeline Config Layout

This directory contains Hydra config groups used by training and inference entrypoints.

## Root Pipeline Configs

- `classifier_pipeline.yaml`: classifier training defaults and Hydra runtime settings.
- `regressor_pipeline.yaml`: regressor training defaults and Hydra runtime settings.

## Config Groups

- `ablation/`: shared feature-ablation definitions and defaults.
- `data/`: dataset source and column wiring.
- `target/`: target builder selection.
- `split/`: train/validation split strategy.
- `cv/`: cross-validation setup.
- `model/`: model family + wrapper parameters.
- `hparams_search/`: Optuna/Hydra search spaces.
- `experiment/`: reusable sweep presets and run metadata.

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
