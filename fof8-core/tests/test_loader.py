from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from fof8_core.loader import FOF8Loader


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
