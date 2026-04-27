# fof8-scout / fof8-gen

A Python toolkit for automating FOF8 season simulation and exporting raw draft data for use in the scouting pipeline.

## Project Overview

This module handles the **Collection** phase of the scouting pipeline. It uses RPA (Robotic Process Automation) to simulate decades of football seasons and export raw CSV snapshots from Front Office Football 8.

### Directory Structure
- `data/raw/`: Unprocessed CSV exports (Git ignored).
- `src/fof8_gen/`: The automation and snapshotting logic.
- `src/fof8_gen/resources/images/`: UI screenshots for image recognition.

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
    uv sync --package fof8-gen
    ```

---

## Quick Start

To begin the automated data collection process:

1. Create a league folder in `data/raw/<LEAGUE_NAME>/`.
2. Copy `templates/metadata.example.yaml` to that folder, rename it to `metadata.yaml`, and update the settings.
3. Open FOF8 to the desired league, ensuring you are on the "Begin Training Camp" screen.
4. Run the automation:
   ```powershell
   uv run gather-data --metadata data/raw/DRAFT001/metadata.yaml --iterations 100
   ```
5. Switch back to the FOF8 window within 5 seconds.

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
