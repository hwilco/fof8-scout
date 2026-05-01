"""Award-count target builders sourced from end-of-simulation award snapshots."""

import polars as pl

from fof8_core.loader import FOF8Loader


def get_awards(
    loader: FOF8Loader,
    award_names: list[str] | None = None,
    target_name: str = "Award_Count",
) -> pl.DataFrame:
    """
    Calculates the cumulative number of awards won by each player.

    Args:
        loader: An instance of FOF8Loader.
        award_names: Optional list of specific award names to count.
                     If None, all awards are counted.
                     If a list is provided, only those awards are counted.
        target_name: The name of the resulting count column.

    Returns:
        A Polars DataFrame with (Player_ID, {target_name}).
    """
    with pl.StringCache():
        # awards.csv is cumulative; scanning all years leads to duplicates.
        # We only need the final available snapshot.
        lf_awards = loader.scan_file("awards.csv", year=loader.final_sim_year)

        if award_names is not None:
            lf_awards = lf_awards.filter(pl.col("Award").is_in(award_names))

        return (
            lf_awards.rename({"Player/Coach": "Player_ID"})
            .group_by("Player_ID")
            .agg(pl.len().alias(target_name))
            .collect()
        )
