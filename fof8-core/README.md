# fof8-core

The central engine for the `fof8-scout` monorepo. This package provides shared domain logic, standardized data loading, and highly optimized Polars pipelines for Front Office Football 8 (FOF8) data.

## Role and Capabilities

`fof8-core` serves as the single source of truth for how FOF8 data is interpreted and processed across the repository. It is used by `fof8-gen` for data collection decisions and by `fof8-ml` for machine learning modeling.

- **Data Loading (`FOF8Loader`)**: Multi-year CSV scanning with automatic year injection and memory-efficient schema enforcement.
- **Domain Schemas**: Standardized Polars schemas using downcasting (`Int8`/`Int16`) and categorical types to handle 100+ years of simulation data in memory.
- **Feature Engineering**: Advanced transformations including position-relative Z-scores and scouting uncertainty metrics.
- **Financial Pipelines**: Longitudinal tracking of player contracts and annual salary cap shares for VORP modeling.
- **Leakage Prevention**: Enforced boundaries between pre-season predictive features and post-season career outcomes.

## Installation

This package is a member of the `fof8-scout` uv workspace. It is automatically available to other modules (like `fof8-ml`) when the workspace is synced.

## Usage

```python
import polars as pl
from fof8_core import FOF8Loader, get_draft_class, get_career_outcomes

# Initialize the loader
loader = FOF8Loader(base_path="./data", league_name="DRAFT003")

# Load predictive features for a specific draft class (includes Z-scores & deltas)
features_df = get_draft_class(loader, year=2050)

# Load annual financial data for longitudinal analysis (VORP support)
financials_df = get_annual_financials(loader)
```

## Development

To run the test suite for this package:

```bash
uv run --package fof8-core pytest fof8-core/tests/
```

To run linting and formatting:

```bash
uv run ruff check fof8-core/
uv run ruff format fof8-core/
```
