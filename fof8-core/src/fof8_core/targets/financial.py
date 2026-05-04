"""Financial and contract-based target builders for longitudinal value metrics."""

import polars as pl

from fof8_core.loader import FOF8Loader


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
            lf_records.filter(pl.col("Experience") == 1)
            .select(
                [
                    "Player_ID",
                    "Year",
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
            .with_columns(
                (pl.col("Year") + pl.col("Contract_Year_Index") - 1).alias("Actual_Payout_Year")
            )
            .join(df_caps, left_on="Actual_Payout_Year", right_on="Year", how="left")
            .with_columns(
                ((pl.col("Salary") + pl.col("Bonus")) / pl.col("Cap_10k")).alias(
                    "Annual_Expected_Cap_Share"
                )
            )
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
