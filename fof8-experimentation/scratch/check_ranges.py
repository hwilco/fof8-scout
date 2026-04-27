from pathlib import Path

import polars as pl

data_dir = Path("/workspaces/fof8-scout/data-generation/data/raw/DRAFT003")

# Find the latest year
years = [int(p.name) for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()]
latest_year = max(years)

# Check player_information
info_df = pl.read_csv(data_dir / str(latest_year) / "player_information.csv")
print(f"Max Player_ID: {info_df['Player_ID'].max()}")
print(f"Max Draft_Year: {info_df['Draft_Year'].max()}")
print(f"Max Career_Games: {info_df['Career_Games_Played'].max()}")

# Check rookies (from any year that has it)
rookie_path = data_dir / "2021" / "rookies.csv"
if rookie_path.exists():
    rookie_df = pl.read_csv(rookie_path)
    print(f"Max Grade: {rookie_df['Grade'].max()}")
    print(f"Max Height: {rookie_df['Height'].max()}")
    print(f"Max Weight: {rookie_df['Weight'].max()}")
