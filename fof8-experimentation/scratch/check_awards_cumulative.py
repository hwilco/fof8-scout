import polars as pl
from fof8_core.loader import FOF8Loader
from pathlib import Path

loader = FOF8Loader(base_path="/workspaces/fof8-scout/fof8-gen/data/raw", league_name="DRAFT003")

try:
    lf_awards_2020 = loader.scan_file("awards.csv", year=2020).collect()
    lf_awards_2021 = loader.scan_file("awards.csv", year=2021).collect()
    print("2020 count:", len(lf_awards_2020))
    print("2021 count:", len(lf_awards_2021))
    print("2021 sample year column:", lf_awards_2021["Year"].unique())
except Exception as e:
    print(f"Error: {e}")
