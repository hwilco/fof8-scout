import polars as pl
from fof8_core.loader import FOF8Loader
from fof8_core.features import get_draft_class
from fof8_ml.data.dataset import apply_position_mask

loader = FOF8Loader(base_path="/workspaces/fof8-scout/fof8-gen/data/raw", league_name="DRAFT003")
df = get_draft_class(loader, 2021)

print(f"Original shape: {df.shape}")

# Apply the mask directly for verification
masked_df = apply_position_mask(df)

# Check a QB
qb_data = masked_df.filter(pl.col("Position_Group") == "QB").head(1)
man_defense = qb_data.get_column("Mean_Man-to-Man_Defense")[0]
accuracy = qb_data.get_column("Mean_Accuracy")[0]

print(f"QB Position_Group: {qb_data.get_column('Position_Group')[0]}")
print(f"QB Mean_Man-to-Man_Defense (should be None): {man_defense}")
print(f"QB Mean_Accuracy (should NOT be None): {accuracy}")

# Check a CB
cb_data = masked_df.filter(pl.col("Position_Group") == "CB").head(1)
man_defense_cb = cb_data.get_column("Mean_Man-to-Man_Defense")[0]
accuracy_cb = cb_data.get_column("Mean_Accuracy")[0]

print(f"\nCB Position_Group: {cb_data.get_column('Position_Group')[0]}")
print(f"CB Mean_Man-to-Man_Defense (should NOT be None): {man_defense_cb}")
print(f"CB Mean_Accuracy (should be None): {accuracy_cb}")

# Verify that nulls exist in the masked dataframe
null_counts = masked_df.null_count()
total_nulls = sum(null_counts.row(0))
print(f"\nTotal Nulls in masked dataframe: {total_nulls}")
