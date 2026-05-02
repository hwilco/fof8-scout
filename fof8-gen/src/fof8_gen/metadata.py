"""Metadata loading and validation helpers for generation runs."""

from pathlib import Path


def load_metadata(path: Path):
    """Load and validate metadata YAML, returning (data, league_name)."""
    try:
        import yaml

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
