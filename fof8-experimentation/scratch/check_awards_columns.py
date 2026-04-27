import polars as pl
from fof8_core.loader import FOF8Loader
from pathlib import Path

loader = FOF8Loader(base_path="/workspaces/fof8-scout/fof8-gen/data/raw", league_name="DRAFT003")

try:
    lf_awards_2021 = loader.scan_file("awards.csv", year=2021).collect()
    print("Columns:", lf_awards_2021.columns)
    # Check if there's another year column
    # Actually, scan_file in loader.py does:
    # return lf.with_columns(
    #     pl.col("filepath").str.extract(regex_pattern, 1).cast(pl.Int16).alias("Year")
    # ).drop("filepath")
    # If "Year" already exists, it OVERWRITES it.
except Exception as e:
    print(f"Error: {e}")
