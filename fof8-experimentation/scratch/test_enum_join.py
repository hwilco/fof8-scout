import polars as pl
from fof8_core.schemas import POSITION_GROUPS

# 1. Create a DF with the Enum Position_Group
df_enum = pl.DataFrame({"Year": [2021, 2021], "Position_Group": ["QB", "RB"]}).with_columns(
    pl.col("Position_Group").cast(POSITION_GROUPS)
)

print("Enum DF Schema:", df_enum.schema)

# 2. Create a DF with a String Position_Group (like ironman_counts used to be)
df_str = pl.DataFrame({"Position_Group": ["QB", "RB"], "Value": [1, 2]})

print("String DF Schema:", df_str.schema)

try:
    df_joined = df_enum.join(df_str, on="Position_Group")
    print("Join successful!")
except Exception as e:
    print(f"Join failed: {e}")

# 3. Create a DF with the Enum Position_Group (like ironman_counts is now)
df_enum_ref = pl.DataFrame(
    {"Position_Group": ["QB", "RB"], "Value": [1, 2]},
    schema={"Position_Group": POSITION_GROUPS, "Value": pl.Int64},
)

print("Enum Ref DF Schema:", df_enum_ref.schema)

try:
    df_joined_fixed = df_enum.join(df_enum_ref, on="Position_Group")
    print("Fixed Join successful!")
except Exception as e:
    print(f"Fixed Join failed: {e}")
