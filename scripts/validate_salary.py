"""
Validate DATA_REFERENCE.md claims against actual DRAFT003 data.
Part 2: Salary units check
"""

from pathlib import Path

import polars as pl

data_dir = Path("/workspaces/fof8-scout/data-generation/data/raw/DRAFT003")
years = sorted(int(p.name) for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit())

print("=== SALARY UNITS CHECK ===")
for check_year in [2050, 2121, years[-1]]:
    if check_year in years:
        # Universe info
        uni = pl.read_csv(data_dir / str(check_year) / "universe_info.csv")
        cap_row = uni.filter(pl.col("Information").str.contains("Salary Cap"))
        min_row = uni.filter(pl.col("Information").str.contains("Minimum Salary"))
        print(f"\n  Year {check_year}:")
        if len(cap_row) > 0:
            cap_info_str = cap_row["Information"][0]
            cap_val = int(cap_row.select(pl.col("Value/Round/Position").cast(pl.Int64))[0, 0])
            print(f"    Cap info key: '{cap_info_str}'")
            print(f"    Cap raw value: {cap_val}")
            print(f"    If tens of thousands: ${cap_val * 10_000:,}")
            print(f"    If thousands: ${cap_val * 1_000:,}")
        if len(min_row) > 0:
            min_info_str = min_row["Information"][0]
            min_val = int(min_row.select(pl.col("Value/Round/Position").cast(pl.Int64))[0, 0])
            print(f"    Min salary key: '{min_info_str}'")
            print(f"    Min salary raw value: {min_val}")
            print(f"    If thousands: ${min_val * 1_000:,}")

        # Sample player salaries
        pr = pl.read_csv(data_dir / str(check_year) / "player_record.csv")
        salary_cols = [c for c in pr.columns if c.startswith("Salary_Year")]
        if salary_cols:
            sal_df = pr.select(["Player_ID"] + salary_cols)
            nonzero = sal_df.filter(pl.col(salary_cols[0]) > 0)
            max_sal = int(nonzero.select(pl.col(salary_cols[0]).max())[0, 0])
            mean_sal = float(nonzero.select(pl.col(salary_cols[0]).mean())[0, 0])
            min_sal = int(nonzero.select(pl.col(salary_cols[0]).min())[0, 0])
            print(
                f"    Salary_Year_1 stats (nonzero): "
                f"max={max_sal}, mean={mean_sal:.0f}, min={min_sal}"
            )
            print(
                f"    If salary in thousands: max=${max_sal * 1_000:,}, "
                f"mean=${mean_sal * 1_000:,.0f}, min=${min_sal * 1_000:,}"
            )
            print(
                f"    If salary in tens of thousands: max=${max_sal * 10_000:,}, "
                f"mean=${mean_sal * 10_000:,.0f}, min=${min_sal * 10_000:,}"
            )

            # Show the top 5 paid players
            top5 = nonzero.sort(salary_cols[0], descending=True).head(5)
            print(f"    Top 5 Salary_Year_1 values: {top5[salary_cols[0]].to_list()}")

# Check player_season for real data
print("\n\n=== PLAYER_SEASON STATS CHECK ===")
for check_year in [2050, 2121]:
    path = data_dir / str(check_year) / f"player_season_{check_year}.csv"
    if path.exists():
        ps = pl.read_csv(path, n_rows=50)
        print(f"\n  Year {check_year}:")
        print(f"  Columns (first 15): {ps.columns[:15]}")
        print(f"  Total shape (sample): {ps.shape}")
        # Show a few rows
        id_cols = [c for c in ps.columns if c in ["Player_ID", "Week", "Game_Number"]]
        stat_cols_sample = [
            c
            for c in ps.columns
            if c not in ["Player_ID", "Record_Number", "Year", "Week", "Game_Number"]
        ][:5]
        print("  Sample data:")
        sample = ps.select(id_cols + stat_cols_sample).head(10)
        print(sample)

# Files per year
print("\n\n=== FILES PER YEAR (sample) ===")
for check_year in [2020, 2021, 2050, 2121, years[-1]]:
    if check_year in years:
        files = sorted(f.name for f in (data_dir / str(check_year)).iterdir())
        print(f"  Year {check_year}: {len(files)} files")
        print(f"    {files}")

# Bonus column check
print("\n\n=== BONUS COLUMNS CHECK ===")
pr = pl.read_csv(data_dir / "2050" / "player_record.csv")
bonus_cols = [c for c in pr.columns if c.startswith("Bonus")]
salary_cols = [c for c in pr.columns if c.startswith("Salary")]
print(f"  Salary columns: {salary_cols}")
print(f"  Bonus columns: {bonus_cols}")
nonzero_bonus = pr.filter(pl.col(bonus_cols[0]) > 0)
if len(nonzero_bonus) > 0:
    max_bonus = int(nonzero_bonus.select(pl.col(bonus_cols[0]).max())[0, 0])
    mean_bonus = float(nonzero_bonus.select(pl.col(bonus_cols[0]).mean())[0, 0])
    min_bonus = int(nonzero_bonus.select(pl.col(bonus_cols[0]).min())[0, 0])
    print(
        f"  Bonus_Year_1 stats (nonzero): max={max_bonus}, mean={mean_bonus:.0f}, min={min_bonus}"
    )
