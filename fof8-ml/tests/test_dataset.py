from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from fof8_core.targets.draft_outcomes import DRAFT_OUTCOME_TARGET_COLUMNS
from fof8_ml.data.career_threshold_dataset import build_career_threshold_dataset
from fof8_ml.data.categorical import bucket_rare_colleges, cast_categoricals_to_enum
from fof8_ml.data.economic_dataset import build_economic_dataset


@pytest.fixture
def mock_loader():
    loader = MagicMock()

    # Mock scan_file for different files
    def scan_file_side_effect(filename, year=None):
        if filename == "rookies.csv":
            return pl.LazyFrame(
                [
                    {
                        "Player_ID": 1,
                        "Year": 2024,
                        "Position_Group": "QB",
                        "First_Name": "John",
                        "Last_Name": "Doe",
                        "Draft_Round": 1,
                    },  # Drafted, high peak
                    {
                        "Player_ID": 2,
                        "Year": 2024,
                        "Position_Group": "WR",
                        "First_Name": "Jane",
                        "Last_Name": "Smith",
                        "Draft_Round": 0,
                    },  # Undrafted
                    {
                        "Player_ID": 3,
                        "Year": 2024,
                        "Position_Group": "QB",
                        "First_Name": "Low",
                        "Last_Name": "Peak",
                        "Draft_Round": 5,
                    },  # Drafted, low peak
                ]
            )
        elif filename == "draft_personal.csv":
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024, "High_Accuracy": 50, "Low_Accuracy": 40},
                    {"Player_ID": 2, "Year": 2024, "High_Accuracy": 60, "Low_Accuracy": 50},
                    {"Player_ID": 3, "Year": 2024, "High_Accuracy": 40, "Low_Accuracy": 30},
                ]
            )
        elif filename == "player_information_pre_draft.csv":
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024, "Year_Born": 2002, "Position": "QB"},
                    {"Player_ID": 2, "Year": 2024, "Year_Born": 2002, "Position": "WR"},
                    {"Player_ID": 3, "Year": 2024, "Year_Born": 2001, "Position": "QB"},
                ]
            )
        elif (
            filename == "player_ratings_season_*.csv"
            or filename == "player_ratings_season_2024.csv"
        ):
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 45},
                    {"Player_ID": 2, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 30},
                    {"Player_ID": 3, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 20},
                ]
            )
        elif filename == "player_record.csv":
            # Only players 1 and 3 (drafted) make it into records
            return pl.LazyFrame(
                [
                    {
                        "Player_ID": 1,
                        "Year": 2024,
                        "Position": "QB",
                        "Salary_Year_1": 100,
                        "Salary_Year_2": 100,
                        "Salary_Year_3": 100,
                        "Salary_Year_4": 100,
                        "Salary_Year_5": 100,
                        "Bonus_Year_1": 50,
                        "Bonus_Year_2": 50,
                        "Bonus_Year_3": 50,
                        "Bonus_Year_4": 50,
                        "Bonus_Year_5": 50,
                        "Experience": 1,
                        "S_Games_Started": 16,
                        "S_Pass_Plays": 500,
                        "S_Run_Plays": 0,
                        "S_Special_Teams_Plays": 0,
                    },
                    {
                        "Player_ID": 3,
                        "Year": 2024,
                        "Position": "QB",
                        "Salary_Year_1": 25,
                        "Salary_Year_2": 25,
                        "Salary_Year_3": 25,
                        "Salary_Year_4": 25,
                        "Salary_Year_5": 25,
                        "Bonus_Year_1": 10,
                        "Bonus_Year_2": 10,
                        "Bonus_Year_3": 10,
                        "Bonus_Year_4": 10,
                        "Bonus_Year_5": 10,
                        "Experience": 1,
                        "S_Games_Started": 16,
                        "S_Pass_Plays": 500,
                        "S_Run_Plays": 0,
                        "S_Special_Teams_Plays": 0,
                    },
                ]
            )
        elif filename == "universe_info.csv":
            return pl.LazyFrame(
                [
                    {
                        "Year": 2024,
                        "Information": "Salary Cap (in tens of thousands)",
                        "Value/Round/Position": 20000,
                    },
                ]
            )
        elif filename == "player_information_post_sim.csv":
            # Player 2 is missing (purged) if scanning final year,
            # but our new targets.py logic scans all years.
            return pl.LazyFrame(
                [
                    {
                        "Player_ID": 1,
                        "Year": 2024,
                        "Career_Games_Played": 16,
                        "Championship_Rings": 0,
                        "Hall_of_Fame_Flag": 0,
                        "Number_of_Seasons": 1,
                    },
                    {
                        "Player_ID": 1,
                        "Year": 2025,
                        "Career_Games_Played": 32,  # Increased in latest year
                        "Championship_Rings": 1,
                        "Hall_of_Fame_Flag": 0,
                        "Number_of_Seasons": 2,
                    },
                    {
                        "Player_ID": 2,
                        "Year": 2024,
                        "Career_Games_Played": 0,
                        "Championship_Rings": 0,
                        "Hall_of_Fame_Flag": 0,
                        "Number_of_Seasons": 1,
                    },
                    {
                        "Player_ID": 3,
                        "Year": 2024,
                        "Career_Games_Played": 16,
                        "Championship_Rings": 0,
                        "Hall_of_Fame_Flag": 0,
                        "Number_of_Seasons": 1,
                    },
                    {
                        "Player_ID": 3,
                        "Year": 2025,
                        "Career_Games_Played": 16,  # Did not play any more games in the latest year
                        "Championship_Rings": 0,
                        "Hall_of_Fame_Flag": 0,
                        "Number_of_Seasons": 2,
                    },
                ]
            )
        return pl.LazyFrame([])

    loader.scan_file.side_effect = scan_file_side_effect
    loader.get_active_team_id.return_value = None
    return loader


def test_build_career_threshold_dataset_preserves_undrafted(mock_loader):
    with patch("fof8_ml.data.career_threshold_dataset.FOF8Loader", return_value=mock_loader):
        X, y = build_career_threshold_dataset(
            raw_path="fake",
            league_name="fake",
            year_range=[2024, 2024],
            final_sim_year=2024,
            career_threshold=20,  # Set higher than 2024 value (16) to check 2025 (32) used
        )

        # We expect 3 players (John Doe, Jane Smith, Low Peak)
        assert len(X) == 3
        # John Doe (ID 1) should clear threshold (Games Played 32 >= 20)
        # If it incorrectly picked the 2024 record (50 games), this would fail.
        assert y[0] == 1
        # Jane Smith (ID 2) should not clear threshold (Games Played 0 < 20)
        assert y[1] == 0
        # Low Peak (ID 3) should not clear threshold (Games Played 16 < 20)
        assert y[2] == 0


def test_build_economic_dataset_preserves_undrafted_peak(mock_loader):
    with patch("fof8_ml.data.economic_dataset.FOF8Loader", return_value=mock_loader):
        X, y, metadata = build_economic_dataset(
            raw_path="fake", league_name="fake", year_range=[2024, 2024]
        )

        assert len(X) == 3
        # Jane Smith (ID 2) should have DPO = 0 (because merit is 0)
        # But her target row should exist.
        assert y["Cleared_Sieve"][1] == 0
        assert y["DPO"][1] == 0.0
        assert y["Economic_Success"][1] == 0
        assert y["Positive_DPO"][1] == 0.0
        assert y["Positive_Career_Merit_Cap_Share"][1] == 0.0

        # Target-like derived columns are not model features.
        for col in DRAFT_OUTCOME_TARGET_COLUMNS:
            assert col not in X.columns

        # Verify metadata
        assert metadata["First_Name"][1] == "Jane"
        assert metadata["Last_Name"][1] == "Smith"


def test_bucket_rare_colleges():
    df = pl.DataFrame({"College": ["A", "A", "B", "C"], "n": [1, 2, 3, 4]})
    out = bucket_rare_colleges(df, min_count=2)
    assert out["College"].to_list() == ["A", "A", "Other", "Other"]


def test_cast_categoricals_to_enum():
    df = pl.DataFrame(
        {
            "Position_Group": ["QB", "WR"],
            "College": ["A", "B"],
            "Score": [1, 2],
        }
    ).with_columns(pl.col("Position_Group").cast(pl.Categorical))
    out = cast_categoricals_to_enum(df)
    assert out.schema["Position_Group"] == pl.Enum(["QB", "WR"])
    assert out.schema["College"] == pl.Enum(["A", "B"])
    assert out.schema["Score"] == pl.Int64


if __name__ == "__main__":
    # For manual verification
    import polars as pl

    pl.Config.set_ascii_tables(True)
