## 1. Environment & Execution
- **Crucial:** This project strictly uses `uv` for environment management.
- NEVER suggest or execute global Python commands (like `pip`, `python`, or `pytest` directly).
- ALWAYS prefix execution commands with `uv run`.
  - *Correct:* `uv run dvc repro process_features`
  - *Correct:* `uv run pytest fof8-core/tests`
  - *Incorrect:* `dvc repro` or `pytest`

## 2. Code Generation Standards
- **Type Hinting:** All new functions and methods MUST include complete Python type hints (PEP 484).
- **Docstrings:** Use Google-style docstrings for all classes and functions. Include `Args:` and `Returns:` blocks.
- **Polars over Pandas:** This project uses Polars for all dataframes. Avoid importing or using Pandas.

## 3. The "Done" Definition
Before concluding any feature addition or refactor, you must autonomously complete these steps:
1. Write or update the corresponding unit test in the relevant `tests/` directory.
2. Run `uv run ruff check --fix` and `uv run ruff format` on the modified files.
3. Check if any `README.md` or other documentationrequires updating based on the changes.

## 4. Data Version Control (DVC) Rules
- **The Data Boundary:** Large files (like `.csv`, `.parquet`, or `.joblib` model artifacts) are strictly tracked by DVC. NEVER stage or commit these files using Git.
- **Tracking Data:** If you generate new raw data or a new base artifact, you must track it using `uv run dvc add <file>`.
- **Committing:** Always stage the resulting `.dvc` stub files and the `dvc.lock` file to Git, not the data itself.
- **Pipeline Execution:** If you modify and intend to run code in `fof8-core/loader.py`, `fof8-ml/preprocessing.py`, or update `pipelines/conf/` YAMLs, you must re-run the affected pipeline stages using `uv run dvc repro`. Do not run the python scripts directly unless debugging a specific function.
- **Temporary Data:** If you generate temporary data for testing, add the directory to .dvcignore.
