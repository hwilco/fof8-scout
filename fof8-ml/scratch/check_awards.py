import polars as pl
from fof8_core.loader import FOF8Loader
from pathlib import Path

loader = FOF8Loader(base_path="/workspaces/fof8-scout/fof8-gen/data/raw", league_name="DRAFT003")

try:
    lf_awards = loader.scan_file("awards.csv")
    df_awards_sample = lf_awards.head(5).collect()
    print("Columns:", df_awards_sample.columns)
    print("Sample data:")
    print(df_awards_sample)
except Exception as e:
    print(f"Error: {e}")
