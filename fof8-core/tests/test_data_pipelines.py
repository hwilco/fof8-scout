import polars as pl
import pytest
from fof8_core import get_annual_financials, get_career_outcomes, get_draft_class


def test_get_draft_class_pipeline(mock_loader, tmp_path):
    # Setup mock data files for two years to test globs and joins
    for year in [2020, 2021]:
        year_dir = tmp_path / "DRAFT003" / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # rookies.csv
        pl.DataFrame(
            {
                "Player_ID": [1, 2],
                "Position_Group": ["QB", "QB"],
                "College": ["State", "U"],
                "Dash": [450, 480],  # 4.50, 4.80
                "Strength": [20, 25],
                "Agility": [700, 750],
                "Jump": [30, 35],
            }
        ).write_csv(year_dir / "rookies.csv")

        # draft_personal.csv
        pl.DataFrame(
            {
                "Player_ID": [1, 2],
                "Interviewed": [1, 0],
                "High_Passing": [80, 60],
                "Low_Passing": [70, 40],
            }
        ).write_csv(year_dir / "draft_personal.csv")

        # player_information_pre_draft.csv
        pl.DataFrame(
            {"Player_ID": [1, 2], "Position": ["QB", "QB"], "Year_Born": [1998, 1999]}
        ).write_csv(year_dir / "player_information_pre_draft.csv")

    df = get_draft_class(mock_loader, year=2020)

    assert isinstance(df, pl.DataFrame)
    # 7 original rookie cols + 4 Z-score cols + 2 original personal cols
    # + 1 delta col + 1 Year col = 15
    assert "Dash_Z" in df.columns
    assert "Delta_Passing" in df.columns
    assert df.filter(pl.col("Player_ID") == 1)["Delta_Passing"][0] == 10
    assert df.shape[0] == 2


def test_get_career_outcomes_pipeline(mock_loader, tmp_path):
    year_dir_2143 = tmp_path / "DRAFT003" / "2143"
    year_dir_2143.mkdir(parents=True)
    pl.DataFrame(
        {
            "Player_ID": [1, 2],
            "Career_Games_Played": [100, 10],
            "Number_of_Seasons": [6, 2],
            "Championship_Rings": [1, 0],
            "Hall_of_Fame_Flag": [0, 0],
        }
    ).write_csv(year_dir_2143 / "player_information_post_sim.csv")

    year_dir_2144 = tmp_path / "DRAFT003" / "2144"
    year_dir_2144.mkdir(parents=True)
    pl.DataFrame(
        {
            "Player_ID": [1, 2, 3],
            "Career_Games_Played": [160, 10, None],
            "Number_of_Seasons": [10, 2, None],
            "Championship_Rings": [2, 0, None],
            "Hall_of_Fame_Flag": [1, 0, None],
        }
    ).write_csv(year_dir_2144 / "player_information_post_sim.csv")

    df = get_career_outcomes(mock_loader)

    assert df.shape == (3, 5)
    assert df.columns == [
        "Player_ID",
        "Career_Games_Played",
        "Championship_Rings",
        "Hall_of_Fame_Flag",
        "Number_of_Seasons",
    ]

    # Regression: when scanning all years, each player's latest row is retained.
    assert df.filter(pl.col("Player_ID") == 1)["Career_Games_Played"][0] == 160
    assert df.filter(pl.col("Player_ID") == 1)["Championship_Rings"][0] == 2

    # Regression: null career values are backfilled to zero.
    assert df.filter(pl.col("Player_ID") == 3)["Career_Games_Played"][0] == 0
    assert df.filter(pl.col("Player_ID") == 3)["Number_of_Seasons"][0] == 0
    assert df.filter(pl.col("Player_ID") == 3)["Championship_Rings"][0] == 0
    assert df.filter(pl.col("Player_ID") == 3)["Hall_of_Fame_Flag"][0] == 0


def test_get_annual_financials_pipeline(mock_loader, tmp_path):
    # Setup data for two years
    for year in [2020, 2021]:
        year_dir = tmp_path / "DRAFT003" / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # universe_info.csv
        pl.DataFrame(
            {
                "Information": ["Salary Cap (in tens of thousands)", "Other"],
                "Value/Round/Position": [20000 if year == 2020 else 21000, 0],
            }
        ).write_csv(year_dir / "universe_info.csv")

        # player_record.csv
        pl.DataFrame(
            {
                "Player_ID": [1, 100],
                "Position": ["QB", "SE"],
                "Salary_Year_1": [100, 200],
                "Bonus_Year_1": [50, 10],
            }
        ).write_csv(year_dir / "player_record.csv")

    df = get_annual_financials(mock_loader)

    assert df.shape[0] == 4  # 2 players * 2 years
    # Year 2020, Player 1: (100 + 50) / 20000 = 0.0075
    val_2020 = df.filter((pl.col("Year") == 2020) & (pl.col("Player_ID") == 1))["Annual_Cap_Share"][
        0
    ]
    assert pytest.approx(val_2020) == 0.0075
