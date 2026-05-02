"""
Validate DATA_REFERENCE.md claims against actual DRAFT003 data.
"""

from pathlib import Path

import polars as pl

data_dir = Path("/workspaces/fof8-scout/data-generation/data/raw/DRAFT003")

# 1. Year range & count
years = sorted(int(p.name) for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit())
print("=== YEAR RANGE ===")
print(f"Actual range: {years[0]}–{years[-1]} ({len(years)} snapshots)")
print("Doc says: 2021–2121 (91 snapshots)")

# 2. Which years have rookies.csv?
rookie_years = sorted(y for y in years if (data_dir / str(y) / "rookies.csv").exists())
print("\n=== ROOKIES.CSV AVAILABILITY ===")
print(f"Years with rookies.csv: {rookie_years[0]}–{rookie_years[-1]} ({len(rookie_years)} years)")
print(f"Missing rookies.csv years: {sorted(set(years) - set(rookie_years))}")

# 3. Total rookies count
total_rookies = 0
yearly_rookie_counts = {}
for y in rookie_years:
    df = pl.read_csv(data_dir / str(y) / "rookies.csv", columns=["Player_ID"])
    yearly_rookie_counts[y] = len(df)
    total_rookies += len(df)
print("\n=== ROOKIE COUNTS ===")
print(f"Total rookies across all years: {total_rookies}")
print("Doc says: 76,290")
print(f"Avg per year: {total_rookies / len(rookie_years):.1f}")
print("Doc says avg: ~838")
print(
    f"Min year count: {min(yearly_rookie_counts.values())} "
    f"(year {min(yearly_rookie_counts, key=lambda y: yearly_rookie_counts[y])})"
)
print(
    f"Max year count: {max(yearly_rookie_counts.values())} "
    f"(year {max(yearly_rookie_counts, key=lambda y: yearly_rookie_counts[y])})"
)

# 4. Grade range
all_grades = []
for y in rookie_years:
    df = pl.read_csv(data_dir / str(y) / "rookies.csv", columns=["Grade"])
    all_grades.append(df)
grades_df = pl.concat(all_grades)
print("\n=== GRADE STATS ===")
print(f"Min: {grades_df['Grade'].min()}, Max: {grades_df['Grade'].max()}")
print(f"Mean: {grades_df['Grade'].mean():.2f}, Std: {grades_df['Grade'].std():.2f}")
q = grades_df["Grade"].quantile([0.25, 0.5, 0.75])
print(
    f"Quantiles: {grades_df['Grade'].quantile(0.25)}, "
    f"{grades_df['Grade'].quantile(0.5)}, {grades_df['Grade'].quantile(0.75)}"
)
above_65 = (grades_df["Grade"] > 65).sum() / len(grades_df) * 100
print(f"Above 65: {above_65:.2f}%")

# 5. player_information.csv at specific years
print("\n=== PLAYER_INFORMATION ROW COUNTS ===")
for check_year in [2031, 2050, 2080, 2121, years[-1]]:
    if check_year in years:
        df = pl.read_csv(
            data_dir / str(check_year) / "player_information.csv", columns=["Player_ID"]
        )
        print(f"  Year {check_year}: {len(df)} rows")

# 6. Final snapshot career data — total drafted players
final_year = years[-1]
print(f"\n=== FINAL SNAPSHOT ({final_year}) ===")
info_final = pl.read_csv(
    data_dir / str(final_year) / "player_information.csv",
    columns=["Player_ID", "Draft_Year", "Draft_Round", "Number_of_Seasons"],
)
drafted = info_final.filter(pl.col("Draft_Round") > 0)
print(f"Total players: {len(info_final)}")
print(f"Total drafted: {len(drafted)}")
print(f"Draft year range: {info_final['Draft_Year'].min()}–{info_final['Draft_Year'].max()}")
print("Doc says: 15,647 drafted players, draft years 2031–2121")

# Also check the 2121 snapshot
info_2121 = pl.read_csv(
    data_dir / "2121" / "player_information.csv", columns=["Player_ID", "Draft_Year", "Draft_Round"]
)
drafted_2121 = info_2121.filter(pl.col("Draft_Round") > 0)
print(f"\n2121 snapshot: {len(info_2121)} total, {len(drafted_2121)} drafted")

# 7. player_record.csv row count
print("\n=== PLAYER_RECORD ROW COUNTS (sample) ===")
for check_year in [2050, 2080, 2121, final_year]:
    if check_year in years:
        df = pl.read_csv(data_dir / str(check_year) / "player_record.csv", columns=["Player_ID"])
        print(f"  Year {check_year}: {len(df)} rows")

# 8. Salary analysis — check units
print("\n=== SALARY UNITS CHECK ===")
for check_year in [2050, 2121, final_year]:
    if check_year in years:
        # Universe info
        uni = pl.read_csv(data_dir / str(check_year) / "universe_info.csv")
        cap_row = uni.filter(pl.col("Information").str.contains("Salary Cap"))
        min_row = uni.filter(pl.col("Information").str.contains("Minimum Salary"))
        print(f"\n  Year {check_year}:")
        if len(cap_row) > 0:
            cap_info_str = cap_row["Information"][0]
            cap_val = cap_row.select(pl.col("Value/Round/Position"))[0, 0]
            print(f"    Cap info: '{cap_info_str}' = {cap_val}")
            print(f"    If tens of thousands: ${cap_val * 10_000:,}")
            print(f"    If thousands: ${cap_val * 1_000:,}")
        if len(min_row) > 0:
            min_info_str = min_row["Information"][0]
            min_val = min_row.select(pl.col("Value/Round/Position"))[0, 0]
            print(f"    Min salary info: '{min_info_str}' = {min_val}")
            print(f"    If thousands: ${min_val * 1_000:,}")

        # Sample player salaries
        pr = pl.read_csv(data_dir / str(check_year) / "player_record.csv")
        salary_cols = [c for c in pr.columns if c.startswith("Salary_Year")]
        if salary_cols:
            sal_df = pr.select(["Player_ID"] + salary_cols)
            max_sal = sal_df.select(pl.col(salary_cols[0])).max()[0, 0]
            mean_sal = (
                sal_df.filter(pl.col(salary_cols[0]) > 0)
                .select(pl.col(salary_cols[0]))
                .mean()[0, 0]
            )
            min_nonzero_sal = (
                sal_df.filter(pl.col(salary_cols[0]) > 0).select(pl.col(salary_cols[0])).min()[0, 0]
            )
            print(
                f"    Salary_Year_1 — max: {max_sal}, mean: {mean_sal:.0f}, "
                f"min(nonzero): {min_nonzero_sal}"
            )
            print(
                f"    If in thousands: max=${max_sal * 1_000:,}, min=${min_nonzero_sal * 1_000:,}"
            )
            print(
                f"    If in tens of thousands: max=${max_sal * 10_000:,}, "
                f"min=${min_nonzero_sal * 10_000:,}"
            )

# 9. Check player_season file for real stats vs zeros
print("\n=== PLAYER_SEASON STATS CHECK ===")
for check_year in [2050, 2121]:
    path = data_dir / str(check_year) / f"player_season_{check_year}.csv"
    if path.exists():
        ps = pl.read_csv(path, n_rows=20)
        print(f"\n  Year {check_year}:")
        print(f"  Columns: {ps.columns[:10]}...")
        print(f"  Shape: {ps.shape}")
        # Check if there are non-zero stat values
        stat_cols = [
            c
            for c in ps.columns
            if c not in ["Player_ID", "Record_Number", "Year", "Week", "Game_Number"]
        ]
        if stat_cols:
            nonzero = ps.select(
                [pl.col(c).cast(pl.Int64, strict=False).sum().alias(c) for c in stat_cols[:5]]
            )
            print(f"  Sample stat sums (first 5 cols): {nonzero.to_dicts()}")

# 10. File count per snapshot
print("\n=== FILES PER YEAR (sample) ===")
for check_year in [2020, 2021, 2050, 2121, final_year]:
    if check_year in years:
        files = list((data_dir / str(check_year)).iterdir())
        print(f"  Year {check_year}: {len(files)} files — {sorted(f.name for f in files)}")

# 11. Draft rate
print("\n=== DRAFT RATE ===")
# Use a few sample years
for check_year in [2031, 2050, 2080]:
    rookies = pl.read_csv(data_dir / str(check_year) / "rookies.csv", columns=["Player_ID"])
    rookie_ids = set(rookies["Player_ID"].to_list())
    # Check how many appear in next year's player_record
    next_year = check_year + 1
    if next_year in years:
        info_next = pl.read_csv(
            data_dir / str(next_year) / "player_information.csv",
            columns=["Player_ID", "Draft_Round", "Draft_Year"],
        )
        drafted_this_class = info_next.filter(
            (pl.col("Draft_Year") == check_year) & (pl.col("Draft_Round") > 0)
        )
        print(
            f"  Year {check_year}: {len(rookies)} rookies, {len(drafted_this_class)} drafted = "
            f"{len(drafted_this_class) / len(rookies) * 100:.1f}%"
        )
