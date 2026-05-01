"""Career outcome target builders derived from post-simulation player info."""

import polars as pl

from fof8_core.loader import FOF8Loader


def get_career_outcomes(loader: FOF8Loader) -> pl.DataFrame:
    """
    Extracts career-long outcome metrics by scanning all available snapshots.
    Using all snapshots ensures we recover players who were purged from the
    game's final cumulative files (e.g., undrafted rookies).

    Args:
        loader: An instance of FOF8Loader.

    Returns:
        A Polars DataFrame containing career targets (Games Played, Rings, HOF).
    """
    with pl.StringCache():
        # Scan ALL years to recover players who were purged from the final cumulative file.
        # We sort by Year and take the last entry for each player to capture their
        # final career state before retirement or purging.
        lf_info = loader.scan_file("player_information_post_sim.csv")

        return (
            lf_info.select(
                [
                    "Player_ID",
                    "Year",
                    "Career_Games_Played",
                    "Championship_Rings",
                    "Hall_of_Fame_Flag",
                    "Number_of_Seasons",
                ]
            )
            .sort("Year")
            .group_by("Player_ID")
            .last()
            .with_columns(
                [
                    pl.col("Career_Games_Played").fill_null(0),
                    pl.col("Championship_Rings").fill_null(0),
                    pl.col("Hall_of_Fame_Flag").fill_null(0),
                    pl.col("Number_of_Seasons").fill_null(0),
                ]
            )
            .drop("Year")
            .collect()
        )
