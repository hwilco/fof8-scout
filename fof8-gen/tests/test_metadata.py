from pathlib import Path

import pytest
from fof8_gen.metadata import clone_metadata_for_league, load_metadata, parse_universe_range


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


def test_parse_universe_range_success():
    assert parse_universe_range("DRAFT009:DRAFT014") == [
        "DRAFT009",
        "DRAFT010",
        "DRAFT011",
        "DRAFT012",
        "DRAFT013",
        "DRAFT014",
    ]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("DRAFT009:TEST014", "prefixes"),
        ("DRAFT09:DRAFT014", "padding"),
        ("DRAFT014:DRAFT009", "ascending"),
    ],
)
def test_parse_universe_range_rejects_invalid_ranges(value: str, message: str):
    with pytest.raises(ValueError, match=message):
        parse_universe_range(value)


def test_clone_metadata_for_league_writes_new_metadata_without_mutating_template(tmp_path: Path):
    base_metadata = tmp_path / "base_metadata.yaml"
    base_metadata.write_text(
        "game_version: FOF8\n"
        "new_game_options:\n"
        "  league_name: TEMPLATE\n"
        "  coach_names_file: false\n"
        "single_player_settings:\n"
        "  injury_setting: 50\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "DRAFT009"
    generated_path = clone_metadata_for_league(base_metadata, "DRAFT009", output_dir)

    generated_data, generated_league = load_metadata(generated_path)
    source_data, source_league = load_metadata(base_metadata)

    assert generated_path == output_dir / "metadata.yaml"
    assert generated_league == "DRAFT009"
    assert generated_data["new_game_options"]["league_name"] == "DRAFT009"
    assert generated_data["new_game_options"]["coach_names_file"] is False
    assert generated_data["single_player_settings"]["injury_setting"] == 50
    assert source_league == "TEMPLATE"
    assert source_data["new_game_options"]["league_name"] == "TEMPLATE"
