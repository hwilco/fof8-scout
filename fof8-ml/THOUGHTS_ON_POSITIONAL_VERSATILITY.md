"Copying" the prospects at inference time is actually the exact right way to handle this. In sports analytics, this is often called Positional Permutation or Cross-Positional Evaluation, and it is one of the biggest competitive advantages you can build into a draft board.

Because your model relies heavily on Position_Group to branch its decision trees, simply changing that text label and re-running the row forces the model to evaluate the player's physical and technical traits against a completely different archetype.

Here is how you can architect this in your inference.py script:

1. Define the FOF8 Positional Transition Matrix
First, you define the realistic transitions based on the FOF8 engine. In FOF8, players generally move "inside" or "down" the defensive spectrum seamlessly, but rarely the other way.

Python
# Valid FOF8 transitions (Original Position -> [Allowed Positions])
TRANSITION_MAP = {
    "T": ["T", "G", "C"],
    "G": ["G", "C"],
    "C": ["C"],
    "DE": ["DE", "DT"],
    "DT": ["DT"],
    "CB": ["CB", "S"],
    "S": ["S"],
    "OLB": ["OLB", "ILB"],
    "ILB": ["ILB"],
    # Skill positions usually stay fixed in the draft pool
    "QB": ["QB"],
    "RB": ["RB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "K": ["K"],
    "P": ["P"]
}
2. Expand the DataFrame (The "Copying" Phase)
Before you run your data through the apply_position_mask logic or categorical encoding, you "explode" the DataFrame so that a Tackle generates three distinct rows (one T, one G, one C), keeping the Player_ID identical.

Using Polars, you can map the transitions and explode the dataset:

```Python
import polars as pl

# 1. Map the original position to a list of allowed positions
df_expanded = df_features.with_columns(
    pl.col("Position_Group").replace_strict(TRANSITION_MAP).alias("Evaluated_Position")
)

# 2. Explode the lists so each player gets a separate row for each allowed position
df_expanded = df_expanded.explode("Evaluated_Position")

# 3. Swap the target column so the model sees the new position
df_expanded = df_expanded.with_columns(
    pl.col("Evaluated_Position").alias("Position_Group")
)
```

3. Re-Apply the Positional Masks
This is the most critical step. In your pipeline configuration, you have mask_positional_features: true.

If a Tackle is being evaluated as a Center, the model needs the Center-specific masking rules applied (e.g., nulling out perimeter run-blocking stats that only Tackles need). Because you exploded the dataset, you can now safely run your apply_position_mask(df_expanded) function, and it will correctly mask the new "copied" rows based on their updated Position_Group.

4. Aggregate and Extract the "Projected Position"
After passing the expanded DataFrame through both the Sieve and the Intensity Regressor, you will have multiple Universal_EDV scores for players with positional flexibility.

To create your final draft board, you group by Player_ID and find the permutation that maximizes their value:

```Python
# Assuming df_results has [Player_ID, Original_Position, Evaluated_Position, Universal_EDV]

# Find the row with the max EDV for each player
df_final_board = df_results.sort("Universal_EDV", descending=True).unique(
    subset=["Player_ID"], keep="first"
)

# Calculate Positional Value gained by switching
df_base_value = df_results.filter(pl.col("Original_Position") == pl.col("Evaluated_Position"))

# Join base value back to see the delta
df_final_board = df_final_board.join(
    df_base_value.select(["Player_ID", "Universal_EDV"]),
    on="Player_ID",
    suffix="_Base"
)

df_final_board = df_final_board.with_columns(
    (pl.col("Universal_EDV") - pl.col("Universal_EDV_Base")).alias("Positional_Flex_Bonus")
)
```
The Output Analysis
By implementing this, your final CSV will tell a massive story. You might see a player like:

Name: John Smith

Drafted Position: T

Projected Position: C

EDV as T: 25.4

EDV as C: 68.2

Positional Flex Bonus: +42.8

This completely changes how you view the draft class. You are no longer just looking for the best Tackles; you are using the AI to actively scout hidden elite Centers masquerading as average Tackles in the college pool.

# On the issue of the new position potentially being a bad fit based on size, and how the new size may be OOD for the trained model at the new position.

This is a classic Out-Of-Distribution (OOD) problem in machine learning. When you feed a tree-based model a data point that exists completely outside the boundaries of its training space, you have to understand exactly how the underlying algorithm handles the math.

Here is how your specific CatBoost pipeline will handle a 290-pound Defensive End being evaluated as an Outside Linebacker:

1. The Terminal Leaf Node (Extrapolation in Trees)
Unlike Linear Regression or Deep Learning, which attempt to mathematically extrapolate lines into infinity (often resulting in wild, exploding predictions for OOD data), decision trees are step functions. They cannot predict a value they have never seen.

If your training data for OLBs had a maximum weight of 265 lbs, CatBoost likely created a final split that says: If Weight > 255 -> Apply Bust Penalty.
When your 290-pound prospect hits that tree, they simply fall into the furthest terminal leaf node. The model doesn't penalize them extra for being 290; it just applies the maximum penalty it learned for the heaviest OLBs in the training set.

2. The Z-Score Trap (Crucial Implementation Detail)
Because trees cap their penalties at the terminal leaf, your biggest defense against these unnatural transitions is the standardized athletic features you engineered (Dash_Z, Agility_Z, Strength_Z).

However, this creates a massive trap for your inference.py script. If a 260-pound DE runs a 4.80 forty-yard dash, his Dash_Z relative to other DEs might be +1.5 (very fast).
If you copy that row, change Position_Group to OLB, but keep the Dash_Z at +1.5, the model will think you just handed it a massive, blazingly fast OLB and might project him as a Hall of Famer.

The Fix: During your expansion phase, you must dynamically recalculate the Z-scores for the "copied" rows using the historical mean and standard deviation of the new target position. A 4.80 forty is fast for a DE, but re-standardized against OLBs, that Dash_Z drops to -2.0. The model will instantly recognize the player is far too slow for the perimeter.

3. The Stage 1 Guillotine
Once the Z-scores are correctly re-calculated, your two-stage architecture perfectly handles the absurdity of the projection.

The Stage 1 Sieve will see the terrible positional Z-scores and output a P_Sieve (career-threshold probability) of something like 0.001. Even if the Stage 2 Intensity model gets confused by the OOD weight and spits out an artificially high expected intensity (default target: `Positive_Career_Merit_Cap_Share`; composite baseline option: `DPO = Peak_Overall * Career_Merit_Cap_Share`), the final Universal_EDV calculation will crush their value down to zero, naturally burying them at the bottom of the draft board for that position.

4. Bounding the Transitions (The Compute Saver)
While the model will mathematically resolve the issue and reject the player, evaluating impossible FOF8 position changes wastes inference compute and litters your draft board with noise.

You can modify your inference script to enforce hard FOF8 mechanical boundaries inside the transition logic using Polars:

Python
# Create conditional rules rather than a blind map
df_expanded = df_features.with_columns([
    pl.when(pl.col("Position_Group") == "DE")
      .then(pl.col("Weight") <= 270) # Only allow DE -> OLB if weight is 270 or below
      .otherwise(False)
      .alias("Can_Play_OLB")
])
By applying these heuristic filters before the model ever sees the data, you ensure the Sieve is only scoring realistic FOF8 counterfactuals.
