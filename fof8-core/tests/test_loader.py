"""Tests for FOF8 loader behaviors, including active team resolution paths."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from fof8_core.loader import FOF8Loader
from fof8_core.loader_team_resolution import TEAM_NAME_TO_ID


def test_loader_init_validation(tmp_path):
    # Should fail if league dir doesn't exist
    with pytest.raises(FileNotFoundError):
        FOF8Loader(tmp_path, league_name="NONEXISTENT")

    # Should succeed if it exists
    (tmp_path / "DRAFT003").mkdir()
    loader = FOF8Loader(tmp_path, league_name="DRAFT003")
    assert loader.league_name == "DRAFT003"


@patch("polars.scan_csv")
def test_scan_file_year_specific(mock_scan, mock_loader, temp_league_dir):
    # Mock return value for scan_csv
    mock_lf = MagicMock(spec=pl.LazyFrame)
    mock_lf.with_columns.return_value = mock_lf
    mock_lf.drop.return_value = mock_lf
    mock_scan.return_value = mock_lf

    filename = "rookies.csv"
    year = 2020

    # Create dummy file so glob works
    expected_path = temp_league_dir / "DRAFT003" / str(year) / filename
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.touch()

    mock_loader.scan_file(filename, year=year)

    # Verify scan_csv was called with the correct path
    expected_path = temp_league_dir / "DRAFT003" / "2020" / filename
    mock_scan.assert_called_once()
    assert str(mock_scan.call_args[0][0]) == str(expected_path)
    assert "schema_overrides" in mock_scan.call_args.kwargs
    assert "dtypes" not in mock_scan.call_args.kwargs


@patch("polars.scan_csv")
def test_scan_file_glob(mock_scan, mock_loader, temp_league_dir):
    mock_lf = MagicMock(spec=pl.LazyFrame)
    mock_lf.with_columns.return_value = mock_lf
    mock_lf.drop.return_value = mock_lf
    mock_scan.return_value = mock_lf

    # Create dummy files so glob works
    year = 2020
    expected_path = temp_league_dir / "DRAFT003" / str(year) / "rookies.csv"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.touch()

    mock_loader.scan_file("rookies.csv")

    expected_path = temp_league_dir / "DRAFT003" / "2020" / "rookies.csv"
    mock_scan.assert_called_once()
    assert str(mock_scan.call_args[0][0]) == str(expected_path)


def test_get_salary_cap(mock_loader, temp_league_dir):
    # Create a dummy universe_info.csv
    year = 2020
    csv_path = temp_league_dir / "DRAFT003" / str(year) / "universe_info.csv"

    df = pl.DataFrame(
        {
            "Information": ["Salary Cap (in tens of thousands)", "Minimum Salary (in thousands)"],
            "Value/Round/Position": [20000, 60],
            "Value/Team/Position": [0, 0],
            "Value/Team": [0, 0],
        }
    )
    df.write_csv(csv_path)

    cap = mock_loader.get_salary_cap(year)
    assert cap == 20000 * 10_000


def test_get_active_team_id_explicit_team_id(tmp_path):
    league_dir = tmp_path / "DRAFT003"
    league_dir.mkdir()
    (league_dir / "metadata.yaml").write_text("team: New York Giants\nteam_id: 19\n")

    loader = FOF8Loader(tmp_path, league_name="DRAFT003")
    assert loader.get_active_team_id() == 19


def test_get_active_team_id_known_team_name(tmp_path):
    league_dir = tmp_path / "DRAFT003"
    league_dir.mkdir()
    (league_dir / "metadata.yaml").write_text("team: New York Giants\n")

    loader = FOF8Loader(tmp_path, league_name="DRAFT003")
    assert loader.get_active_team_id() == TEAM_NAME_TO_ID["new york giants"]


def test_get_active_team_id_missing_metadata(tmp_path):
    league_dir = tmp_path / "DRAFT003"
    league_dir.mkdir()

    loader = FOF8Loader(tmp_path, league_name="DRAFT003")
    assert loader.get_active_team_id() is None


def test_get_active_team_id_team_information_fallback(tmp_path):
    league_dir = tmp_path / "DRAFT003"
    (league_dir / "2020").mkdir(parents=True)
    (league_dir / "metadata.yaml").write_text("team: London Monarchs\n")
    pl.DataFrame(
        {
            "Team": [42, 7],
            "Home_City": ["London", "Dallas"],
            "Team_Name": ["Monarchs", "Cowboys"],
        }
    ).write_csv(league_dir / "2020" / "team_information.csv")

    loader = FOF8Loader(tmp_path, league_name="DRAFT003")
    assert loader.get_active_team_id() == 42


@patch("polars.scan_csv")
def test_scan_file_applies_schema_overrides_during_scan(mock_scan, mock_loader, temp_league_dir):
    mock_lf = MagicMock(spec=pl.LazyFrame)
    mock_lf.with_columns.return_value = mock_lf
    mock_lf.drop.return_value = mock_lf
    mock_lf.rename.return_value = mock_lf
    mock_lf.collect_schema.return_value.names.side_effect = [
        ["Player_IDPlayer_ID", "Season_Statistics_-_Injury_Type", "Future_Overall"],
        ["Player_ID", "Injury_Type", "Future_Overall"],
    ]
    mock_scan.return_value = mock_lf

    year = 2020
    csv_path = temp_league_dir / "DRAFT003" / str(year) / "player_record.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("Player_IDPlayer_ID,Season_Statistics_-_Injury_Type,Future_Overall\n")

    mock_loader.scan_file("player_record.csv", year=year)

    scan_overrides = mock_scan.call_args.kwargs["schema_overrides"]
    assert scan_overrides["Player_IDPlayer_ID"] == pl.Int32
    assert scan_overrides["Season_Statistics_-_Injury_Type"] == pl.String
