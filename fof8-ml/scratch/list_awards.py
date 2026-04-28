import polars as pl
from fof8_core.loader import FOF8Loader
from pathlib import Path

loader = FOF8Loader(base_path="/workspaces/fof8-scout/fof8-gen/data/raw", league_name="DRAFT003")

try:
    lf_awards = loader.scan_file("awards.csv")
    df_awards_all = lf_awards.select("Award").unique().collect()
    print("Unique Awards:")
    print(df_awards_all.sort("Award"))
except Exception as e:
    print(f"Error: {e}")
