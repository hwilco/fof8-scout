import polars as pl
from fof8_core.loader import FOF8Loader
from fof8_core.targets import get_awards
from fof8_core.awards import ALL_LEAGUE_FIRST_TEAM, MAJOR_AWARDS
from pathlib import Path

loader = FOF8Loader(base_path="/workspaces/fof8-scout/fof8-gen/data/raw", league_name="DRAFT003")

print("--- Testing get_awards (all awards) ---")
df_all = get_awards(loader)
print(f"Total players with awards: {len(df_all)}")
print(df_all.sort("Award_Count", descending=True).head(5))

print("\n--- Testing get_awards (Major Awards) ---")
df_major = get_awards(loader, award_names=MAJOR_AWARDS, target_name="Major_Award_Count")
print(f"Total players with major awards: {len(df_major)}")
print(df_major.sort("Major_Award_Count", descending=True).head(5))

print("\n--- Testing get_awards (All-League First Team) ---")
df_allpro = get_awards(loader, award_names=ALL_LEAGUE_FIRST_TEAM, target_name="All_League_Count")
print(f"Total players with All-League First Team honors: {len(df_allpro)}")
print(df_allpro.sort("All_League_Count", descending=True).head(5))

# Verification of a specific player if possible
# Let's see if we can find a player with multiple awards
if len(df_all) > 0:
    top_player_id = df_all.sort("Award_Count", descending=True)["Player_ID"][0]
    print(f"\nChecking awards for Player_ID {top_player_id}:")
    with pl.StringCache():
        lf_awards = loader.scan_file("awards.csv")
        df_player_awards = lf_awards.filter(pl.col("Player/Coach") == top_player_id).collect()
        print(df_player_awards.select(["Year", "Award"]))
