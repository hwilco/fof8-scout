import polars as pl
from fof8_core.loader import FOF8Loader
from fof8_core.features import get_draft_class

loader = FOF8Loader(base_path="/workspaces/fof8-scout/fof8-gen/data/raw", league_name="DRAFT003")
df = get_draft_class(loader, 2021)
print(df.columns)
