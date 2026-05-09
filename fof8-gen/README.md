# fof8-scout / fof8-gen

A Python toolkit for automating FOF8 season simulation and exporting raw draft data for use in the scouting pipeline.

## Project Overview

This module handles the **Collection** phase of the scouting pipeline. It uses RPA (Robotic Process Automation) to simulate decades of football seasons and export raw CSV snapshots from Front Office Football 8.

### Directory Structure
- `data/raw/`: Unprocessed CSV exports (Git ignored).
- `src/fof8_gen/`: The automation and snapshotting logic.
- `src/fof8_gen/resources/images/`: UI screenshots for image recognition.

### Automation Architecture
- `src/fof8_gen/automation.py`: Thin CLI entrypoint used by `gather-data`.
- `src/fof8_gen/automation_runner.py`: High-level run/snapshot orchestration.
- `src/fof8_gen/workflows.py`: Named season-step methods for season simulation and new-universe setup.
- `src/fof8_gen/screen.py`: Image wait/click helpers and sleep-prevention context manager.
- `src/fof8_gen/metadata.py`: Metadata YAML loading, cloning, and universe-range parsing.

---

## Installation & Setup

1.  **Prerequisites**:
    - Front Office Football 8
    - Windows OS (Required for RPA automation)
    - Python 3.12.x
    - [uv](https://docs.astral.sh/uv/)

> [!IMPORTANT]
> **Host Execution Required**: The data gathering automation must be run on your Windows host machine to interact with the game. It will not work inside a Dev Container or a headless environment.

2.  **Install Dependencies** (from the monorepo root):
    ```powershell
    uv sync --all-packages --group dev
    ```

    If you only want the generation package runtime and not the broader workspace tooling, this also works:
    ```powershell
    uv sync --package fof8-gen
    ```

---

## Quick Start

`gather-data` supports two automation modes:

- Existing universe mode: start from an already-created league on the `Begin Training Camp` screen and run the standard export/sim loop.
- Multi-universe generation mode: clone one metadata template into multiple league folders, create each new league in FOF8, complete the initial staff draft, then run the standard export/sim loop for each generated universe.

### Existing Universe Mode

To begin automated collection for an existing league:

1. Create a league folder in `data/raw/<LEAGUE_NAME>/`.
2. Copy `templates/metadata.example.yaml` to that folder, rename it to `metadata.yaml`, and update the settings.
3. Open FOF8 to the desired league, ensuring you are on the "Begin Training Camp" screen.
4. Run the automation:
   ```powershell
   uv run gather-data --metadata data/raw/DRAFT001/metadata.yaml --iterations 51
   ```
5. Switch back to the FOF8 window within 5 seconds.

### Multi-Universe Generation Mode

Use this mode when you want to create several fresh universes from one template and then collect multiple simulated draft histories.

1. Start from a metadata template whose `new_game_options` match the in-game new-game defaults you want to reuse.
2. Set `new_game_options.league_name` in the template to any placeholder value. It will be replaced for each generated universe.
3. Ensure `new_game_options.coach_names_file` is `false`. That alternate branch is not automated yet.
4. Open FOF8 on a screen where `New Game` is accessible.
5. Run:
   ```powershell
   uv run gather-data `
     --metadata templates/metadata.example.yaml `
     --generate-universes DRAFT009:DRAFT014 `
     --output-root data/raw `
     --iterations 51
   ```
6. Switch back to the FOF8 window within 5 seconds.

This creates:

- `data/raw/DRAFT009/metadata.yaml`
- `data/raw/DRAFT010/metadata.yaml`
- `...`
- `data/raw/DRAFT014/metadata.yaml`

Each generated metadata file has `new_game_options.league_name` rewritten to the generated universe name.

### CLI Options

Common options:

- `--metadata PATH`: base metadata file to load.
- `--iterations N`: number of season-loop iterations to run.
- `--fof8-dir DIR`: override the host FOF8 data directory.

Existing-universe options:

- `--output-dir DIR`: override the snapshot output directory for a single existing league.
- `--snapshot-only`: export data once without advancing the season.

Multi-universe options:

- `--generate-universes START:END`: inclusive range like `DRAFT009:DRAFT014`.
- `--output-root DIR`: parent directory for generated universe folders. Defaults to the metadata file parent.
- `--overwrite-metadata`: allow generated `metadata.yaml` files to be replaced.

`--snapshot-only` and `--generate-universes` are mutually exclusive.

### Screenshot Assets Required For Multi-Universe Mode

The generation workflow depends on these image assets under `src/fof8_gen/resources/images/`:

- `save_btn.PNG`
- `new_game_btn.PNG`
- `submit_btn.PNG`
- `league_name_box.PNG`
- `have_scout_finish_draft_btn.PNG`
- `begin_training_camp_btn.PNG`
- `fastest_option.PNG`

Run `uv run gather-data --help` to see all available configuration options.

## Data Versioning (DVC)

After collecting new data, use DVC to version the `data/raw` directory and push it to the remote so it can be pulled into the experimentation environment.

```powershell
# Add/update the data in DVC tracking
uv run dvc commit data/raw.dvc

# Push to the remote
uv run dvc push
```

---

## Development

This project uses **[Ruff](https://docs.astral.sh/ruff/)** for linting and formatting.

```powershell
# Check and auto-fix code
uv run ruff check --fix

# Format code
uv run ruff format
```
