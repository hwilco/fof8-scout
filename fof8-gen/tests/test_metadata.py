from pathlib import Path

import pytest
from fof8_gen.metadata import load_metadata


def test_load_metadata_success(tmp_path: Path):
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text("new_game_options:\n  league_name: Test League\n", encoding="utf-8")

    data, league_name = load_metadata(metadata)

    assert data["new_game_options"]["league_name"] == "Test League"
    assert league_name == "Test League"


def test_load_metadata_empty_file(tmp_path: Path):
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Metadata file is empty"):
        load_metadata(metadata)


def test_load_metadata_missing_league_name(tmp_path: Path):
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text("new_game_options: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="league_name"):
        load_metadata(metadata)


def test_load_metadata_malformed_yaml(tmp_path: Path):
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text("new_game_options:\n  league_name: [oops\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed"):
        load_metadata(metadata)
