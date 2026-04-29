import polars as pl

# TODO: Implement a "Target Registry" pattern here.
# Instead of hardcoding target names (like 'get_career_outcomes'), we should be able to dynamically
# register and resolve target extractors based on config strings
# (e.g., `TARGET_REGISTRY["career_value"]()`).
from .loader import FOF8Loader
from .targets_replacements import ReplacementStrategy, strategy_hybrid_baseline


def get_career_outcomes(loader: FOF8Loader, final_year: int) -> pl.DataFrame:
    """
    Extracts career-long outcome metrics from the final available snapshot.

    Args:
        loader: An instance of FOF8Loader.
        final_year: The final year of the simulation to pull career totals from.

    Returns:
        A Polars DataFrame containing career targets (Games Played, Rings, HOF).
    """
    with pl.StringCache():
        lf_info = loader.scan_file("player_information_post_sim.csv", year=final_year)

        return (
            lf_info.select(
                [
                    "Player_ID",
                    "Draft_Year",
                    "Year_Born",
                    "Draft_Round",
                    "Career_Games_Played",
                    "Championship_Rings",
                    "Hall_of_Fame_Flag",
                    "Number_of_Seasons",
                ]
            )
            .with_columns(
                [
                    (pl.col("Draft_Round") > 0).alias("Was_Drafted"),
                    pl.col("Career_Games_Played").fill_null(0),
                    pl.col("Championship_Rings").fill_null(0),
                    pl.col("Hall_of_Fame_Flag").fill_null(0),
                    pl.col("Number_of_Seasons").fill_null(0),
                ]
            )
            .collect()
        )


def get_annual_financials(loader: FOF8Loader) -> pl.DataFrame:
    """
    Processes longitudinal financial data across all simulation years.
    Useful for calculating VORP and career earnings (Cap Share).

    Args:
        loader: An instance of FOF8Loader.

    Returns:
        A Polars DataFrame with (Player_ID, Year, Position, Annual_Cap_Share).
    """
    with pl.StringCache():
        # 1. Fetch Salary Caps for all years
        lf_caps = loader.scan_file("universe_info.csv")
        df_caps = (
            lf_caps.filter(pl.col("Information") == "Salary Cap (in tens of thousands)")
            .select(["Year", pl.col("Value/Round/Position").cast(pl.Int32).alias("Cap_10k")])
            .collect()
        )

        # 2. Process all player records
        lf_records = loader.scan_file("player_record.csv")

        return (
            lf_records.select(
                [
                    "Player_ID",
                    "Year",
                    "Position",
                    "Salary_Year_1",
                    "Bonus_Year_1",
                ]
            )
            .join(df_caps.lazy(), on="Year", how="left")
            .with_columns(
                ((pl.col("Salary_Year_1") + pl.col("Bonus_Year_1")) / pl.col("Cap_10k")).alias(
                    "Annual_Cap_Share"
                )
            )
            .select(["Player_ID", "Year", "Position", "Annual_Cap_Share"])
            .collect()
        )


def get_peak_overall(loader: FOF8Loader, k: int = 3) -> pl.DataFrame:
    """
    Calculates the mean of the top k Current_Overall values for each player,
    using the league-scout post-exhibition ratings.
    """
    with pl.StringCache():
        lf_ratings = loader.scan_file("player_ratings_season_*.csv")

        return (
            lf_ratings.filter(pl.col("Scouting") == "Exhibition")
            .select(["Player_ID", "Current_Overall"])
            .group_by("Player_ID")
            .agg(pl.col("Current_Overall").top_k(k).mean().alias("Peak_Overall"))
            .collect()
        )


def get_career_vorp(loader: FOF8Loader) -> pl.DataFrame:
    """
    Calculates Playtime-Weighted Market VORP to bypass the rookie wage scale.
    Value is awarded based on starts multiplied by the positional market premium.
    """
    with pl.StringCache():
        # 1. Fetch Salary Caps
        lf_caps = loader.scan_file("universe_info.csv")
        df_caps = lf_caps.filter(
            pl.col("Information") == "Salary Cap (in tens of thousands)"
        ).select(["Year", pl.col("Value/Round/Position").cast(pl.Int32).alias("Cap_10k")])

        # 2. Fetch Player Records & Calculate Actual Cap Shares
        lf_records = loader.scan_file("player_record.csv")
        df_records = (
            lf_records.select(
                [
                    "Player_ID",
                    "Year",
                    "Position",
                    "Salary_Year_1",
                    "Bonus_Year_1",
                    "S_Games_Started",
                ]
            )
            .join(df_caps.lazy(), on="Year", how="left")
            .with_columns(
                [
                    ((pl.col("Salary_Year_1") + pl.col("Bonus_Year_1")) / pl.col("Cap_10k")).alias(
                        "Actual_Cap_Share"
                    ),
                    pl.col("S_Games_Started").max().over("Year").alias("Max_Games"),
                ]
            )
        )

        # 3. Calculate the "Market Rates" per Position per Year
        df_market = (
            df_records.filter(pl.col("Actual_Cap_Share") > 0)
            .with_columns(
                (
                    pl.col("Actual_Cap_Share").rank(descending=True).over(["Year", "Position"])
                    / pl.len().over(["Year", "Position"])
                ).alias("Salary_Rank_Pct")
            )
            .group_by(["Year", "Position"])
            .agg(
                [
                    # The top 25% paid players define the true "Starter Market Rate"
                    pl.col("Actual_Cap_Share")
                    .filter(pl.col("Salary_Rank_Pct") <= 0.25)
                    .mean()
                    .alias("Starter_Rate"),
                    # The 25% to 50% paid players define the "Replacement/Backup Market Rate"
                    pl.col("Actual_Cap_Share")
                    .filter(
                        (pl.col("Salary_Rank_Pct") > 0.25) & (pl.col("Salary_Rank_Pct") <= 0.50)
                    )
                    .mean()
                    .alias("Replacement_Rate"),
                ]
            )
        )

        # 4. Calculate VORP based purely on Starts * Positional Premium
        return (
            df_records.join(df_market, on=["Year", "Position"], how="left")
            .with_columns(
                # Premium = Economic difference between a Starter and a Backup
                (pl.col("Starter_Rate") - pl.col("Replacement_Rate").fill_null(0)).alias(
                    "Positional_Premium"
                )
            )
            .with_columns(
                # VORP = % of season started * Positional Premium
                (
                    (pl.col("S_Games_Started") / pl.col("Max_Games")) * pl.col("Positional_Premium")
                ).alias("Annual_VORP")
            )
            .group_by("Player_ID")
            .agg(pl.col("Annual_VORP").sum().alias("Career_VORP"))
            .collect()
        )


def get_merit_cap_share(loader: FOF8Loader) -> pl.DataFrame:
    """
    Calculates pure merit-based earnings by subtracting the total expected cap share
    of a player's initial rookie contract from their actual career earnings, properly
    accounting for year-over-year cap inflation.
    """
    with pl.StringCache():
        # 1. Get Actual Career Earnings (sum of actual Cap Share per year)
        df_annual = get_annual_financials(loader)
        df_actual_career = df_annual.group_by("Player_ID").agg(
            pl.col("Annual_Cap_Share").sum().alias("Actual_Career_Cap_Share")
        )

        # 2. Get the Salary Cap Lookup Table
        lf_caps = loader.scan_file("universe_info.csv")
        df_caps = (
            lf_caps.filter(pl.col("Information") == "Salary Cap (in tens of thousands)")
            .select(["Year", pl.col("Value/Round/Position").cast(pl.Int32).alias("Cap_10k")])
            .collect()
        )

        # 3. Get Initial Contract Values from their Rookie Season
        lf_records = loader.scan_file("player_record.csv")
        df_rookie_base = (
            lf_records.filter(pl.col("Experience") == 1)  # Isolate rookie year
            .select(
                [
                    "Player_ID",
                    "Year",  # This is the Rookie Year
                    "Salary_Year_1",
                    "Salary_Year_2",
                    "Salary_Year_3",
                    "Salary_Year_4",
                    "Salary_Year_5",
                    "Bonus_Year_1",
                    "Bonus_Year_2",
                    "Bonus_Year_3",
                    "Bonus_Year_4",
                    "Bonus_Year_5",
                ]
            )
            .collect()
        )

        # 4. Unpivot (Melt) the salaries and bonuses to long format to easily match with future caps
        # We handle Salary and Bonus separately, then join them together
        df_salaries = (
            df_rookie_base.unpivot(
                index=["Player_ID", "Year"],
                on=[
                    "Salary_Year_1",
                    "Salary_Year_2",
                    "Salary_Year_3",
                    "Salary_Year_4",
                    "Salary_Year_5",
                ],
                variable_name="Contract_Year_String",
                value_name="Salary",
            )
            .with_columns(
                # Extract the year integer (1-5) from the column name
                pl.col("Contract_Year_String")
                .str.extract(r"(\d+)")
                .cast(pl.Int32)
                .alias("Contract_Year_Index")
            )
            .drop("Contract_Year_String")
        )

        df_bonuses = (
            df_rookie_base.unpivot(
                index=["Player_ID", "Year"],
                on=["Bonus_Year_1", "Bonus_Year_2", "Bonus_Year_3", "Bonus_Year_4", "Bonus_Year_5"],
                variable_name="Contract_Year_String",
                value_name="Bonus",
            )
            .with_columns(
                pl.col("Contract_Year_String")
                .str.extract(r"(\d+)")
                .cast(pl.Int32)
                .alias("Contract_Year_Index")
            )
            .drop("Contract_Year_String")
        )

        # 5. Join Salaries, Bonuses, and the Forward-Looking Cap
        df_rookie_contracts = (
            df_salaries.join(df_bonuses, on=["Player_ID", "Year", "Contract_Year_Index"])
            # Calculate the ACTUAL year this money is paid out
            .with_columns(
                (pl.col("Year") + pl.col("Contract_Year_Index") - 1).alias("Actual_Payout_Year")
            )
            # Join the Cap lookup table on the Actual Payout Year
            .join(df_caps, left_on="Actual_Payout_Year", right_on="Year", how="left")
            # Calculate this specific year's cap share
            .with_columns(
                ((pl.col("Salary") + pl.col("Bonus")) / pl.col("Cap_10k")).alias(
                    "Annual_Expected_Cap_Share"
                )
            )
            # Roll it back up to a single expected value per player
            .group_by("Player_ID")
            .agg(pl.col("Annual_Expected_Cap_Share").sum().alias("Initial_Contract_Cap_Share"))
        )

        # 6. Join to actual career earnings and calculate the final Merit metric
        return (
            df_actual_career.join(df_rookie_contracts, on="Player_ID", how="left")
            .with_columns(pl.col("Initial_Contract_Cap_Share").fill_null(0))
            .with_columns(
                (pl.col("Actual_Career_Cap_Share") - pl.col("Initial_Contract_Cap_Share")).alias(
                    "Career_Merit_Cap_Share"
                )
            )
            .select(["Player_ID", "Career_Merit_Cap_Share"])
        )


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


def calculate_season_av(lf_records: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates a universal Approximate Value (AV) production score for each player-season
    based on raw box-score statistics.
    """
    return (
        lf_records.with_columns(
            [
                # Fill nulls for all season stats to prevent math errors
                pl.col("^S_.*$").fill_null(0).cast(pl.Float32)
            ]
        )
        .with_columns(
            (
                # Passing
                (pl.col("S_Passing_Yards") / 25)
                + (pl.col("S_Touchdown_Passes") * 4)
                - (pl.col("S_Intercepted") * 2)
                - (pl.col("S_Sacked") * 1)
                +
                # Rushing & Receiving
                (pl.col("S_Rushing_Yards") / 10)
                + (pl.col("S_Receiving_Yards") / 10)
                + (pl.col("S_Rushing_Touchdowns") * 6)
                + (pl.col("S_Receiving_Touchdowns") * 6)
                +
                # Defense
                (pl.col("S_Tackles") * 1)
                + (pl.col("S_Assists") * 0.5)
                + ((pl.col("S_Sacks_(x10)") / 10) * 4)
                + (pl.col("S_Interceptions") * 5)
                + (pl.col("S_Passes_Defensed") * 1)
                +
                # Turnovers
                (pl.col("S_Fumbles_Forced") * 3)
                + (pl.col("S_Fumbles_Recovered") * 3)
                - (pl.col("S_Fumbles") * 2)
                +
                # Blocking (Applies to TEs, FBs, WRs, and OL)
                (pl.col("S_Key_Run_Blocks") * 0.5)
                + (pl.col("S_Pancake_Blocks") * 1)
                - (pl.col("S_Sacks_Allowed") * 3)
                +
                # Special Teams / Specialists
                (pl.col("S_Field_Goals_Made") * 3)
                - ((pl.col("S_Field_Goals_Attempted") - pl.col("S_Field_Goals_Made")) * 2)
                + (pl.col("S_Punts_Inside_20") * 1)
                + (pl.col("S_Punt_Returns_Touchdowns") * 6)
                + (pl.col("S_Kick_Return_Touchdowns") * 6)
            ).alias("Base_Stat_AV")
        )
        .with_columns(
            # Offensive Line Proxy: Add start multipliers ONLY for true O-Linemen
            # since they cannot accumulate yardage or tackle stats.
            pl.when(pl.col("Position_Group").is_in(["C", "G", "T"]))
            .then(pl.col("Base_Stat_AV") + (pl.col("S_Games_Started") * 2.5))
            .otherwise(pl.col("Base_Stat_AV"))
            .alias("Season_AV")
        )
    )


def get_career_value_metrics(
    loader: FOF8Loader, strategy: ReplacementStrategy = strategy_hybrid_baseline
) -> pl.DataFrame:
    """
    Calculates universal Approximate Value (AV) and position-adjusted
    Value Over Replacement Player (VORP) using the specified strategy.
    """
    with pl.StringCache():
        lf_records = loader.scan_file("player_record.csv")

        # 1. Calculate Season AV
        lf_av = calculate_season_av(lf_records)

        # 2. Calculate Season-by-Season VORP using the specified strategy
        # Strategies now return the full dataframe with Season_VORP pre-calculated.
        lf_vorp = strategy(lf_av)

        # 3. Aggregate to Career Totals
        return (
            lf_vorp.group_by("Player_ID")
            .agg(
                [
                    pl.col("Season_AV").sum().alias("Career_AV"),
                    pl.col("Season_VORP").sum().alias("Career_VORP"),
                ]
            )
            .collect()
        )
