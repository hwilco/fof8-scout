"""Metadata loading and validation helpers for generation runs."""

import re
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_metadata(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate metadata YAML, returning (data, league_name)."""
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(f"PyYAML is required to load metadata: {e}") from e

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Metadata YAML is malformed: {e}")
    except OSError as e:
        raise RuntimeError(f"Could not read metadata file: {e}")

    if not data:
        raise ValueError("Metadata file is empty")

    league_name = data.get("new_game_options", {}).get("league_name")
    if not league_name:
        raise ValueError("Metadata file is missing 'new_game_options: league_name'")

    return data, league_name


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    """Write metadata YAML to disk, creating the parent directory when needed."""
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(f"PyYAML is required to write metadata: {e}") from e

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Could not serialize metadata YAML: {e}") from e
    except OSError as e:
        raise RuntimeError(f"Could not write metadata file: {e}") from e


def clone_metadata_for_league(
    base_metadata_path: Path,
    league_name: str,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Clone a metadata template for one generated league without mutating the source."""
    data, _ = load_metadata(base_metadata_path)
    cloned = deepcopy(data)
    cloned.setdefault("new_game_options", {})
    cloned["new_game_options"]["league_name"] = league_name

    metadata_path = output_dir / "metadata.yaml"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing metadata file: {metadata_path}. "
            "Pass overwrite=True to replace it."
        )

    write_metadata(metadata_path, cloned)
    return metadata_path


def parse_universe_range(value: str) -> list[str]:
    """Parse an inclusive prefixed numeric universe range like DRAFT009:DRAFT014."""
    match = re.fullmatch(r"([A-Za-z_-]*)(\d+):([A-Za-z_-]*)(\d+)", value.strip())
    if not match:
        raise ValueError(
            "Universe range must look like PREFIX001:PREFIX010 with matching prefixes."
        )

    start_prefix, start_digits, end_prefix, end_digits = match.groups()
    if start_prefix != end_prefix:
        raise ValueError("Universe range prefixes must match.")
    if len(start_digits) != len(end_digits):
        raise ValueError("Universe range numeric padding must match.")

    start_num = int(start_digits)
    end_num = int(end_digits)
    if start_num > end_num:
        raise ValueError("Universe range must be ascending.")

    width = len(start_digits)
    return [f"{start_prefix}{number:0{width}d}" for number in range(start_num, end_num + 1)]
