
PASSING_FEATURES = [
    "Screen_Passes", "Short_Passes", "Medium_Passes", "Long_Passes", "Deep_Passes",
    "Third_Down", "Run_Frequency", "Accuracy", "Timing", "Sense_Rush", "Read_Defense", "Two-Minute_Offense"
]

RUSHING_FEATURES = [
    "Power_Inside", "Third-Down_Runs", "Hole_Recognition", "Elusiveness", "Speed_Outside", "Blitz_Pickup"
]

RECEIVING_FEATURES = [
    "Avoid_Drops", "Get_Downfield", "Route_Running", "Third-Down_Receiving", "Big_Play_Receiving", "Courage", "Adjust_to_Ball"
]

BLOCKING_FEATURES = [
    "Run_Blocking", "Pass_Blocking", "Blocking_Strength"
]

DEFENSIVE_FEATURES = [
    "Run_Defense", "Pass_Rush_Technique", "Man-to-Man_Defense", "Zone_Defense", "Bump-and-Run_Defense",
    "Pass_Rush_Strength", "Play_Diagnosis", "Punishing_Hitter", "Intercepting"
]

KICKING_FEATURES = [
    "Kickoff_Distance", "Kickoff_Hang_Time", "Kicking_Accuracy", "Kicking_Power"
]

PUNTING_FEATURES = [
    "Punting_Power", "Hang_Time", "Directional_Punting", 
]

RETURNER_FEATURES = ["Punt_Returns", "Kick_Returns"]

LONG_SNAPPING_FEATURE = ["Long_Snapping"]

KICK_HOLDING_FEATURE = ["Kick_Holding"]

SPECIAL_TEAMS_FEATURE = ["Special_Teams"]

# Expand to include Mean_ and Delta_ prefixes
def expand_features(base_list):
    expanded = []
    for feat in base_list:
        expanded.append(f"Mean_{feat}")
        expanded.append(f"Delta_{feat}")
    return expanded

# The list of all features that are subject to being masked (nulled)
MASKABLE_FEATURES = expand_features(
    PASSING_FEATURES + RUSHING_FEATURES + RECEIVING_FEATURES + 
    BLOCKING_FEATURES + DEFENSIVE_FEATURES + KICKING_FEATURES + 
    PUNTING_FEATURES + RETURNER_FEATURES + LONG_SNAPPING_FEATURE + 
    KICK_HOLDING_FEATURE + SPECIAL_TEAMS_FEATURE
)

# Explicit map of which features to KEEP for each position group.
# Anything in MASKABLE_FEATURES not in this list will be set to Null for that position.
POSITION_FEATURE_MAP = {
    "QB": expand_features(PASSING_FEATURES + KICK_HOLDING_FEATURE),
    
    "RB": expand_features(RUSHING_FEATURES + RECEIVING_FEATURES + SPECIAL_TEAMS_FEATURE),
    
    "FB": expand_features(RUSHING_FEATURES + RECEIVING_FEATURES + BLOCKING_FEATURES + SPECIAL_TEAMS_FEATURE),
    
    "WR": expand_features(RUSHING_FEATURES + RECEIVING_FEATURES + BLOCKING_FEATURES + RETURNER_FEATURES + SPECIAL_TEAMS_FEATURE),
        
    "TE": expand_features(RUSHING_FEATURES + RECEIVING_FEATURES + BLOCKING_FEATURES + SPECIAL_TEAMS_FEATURE),
    
    "C":  expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "G":  expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "T":  expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    
    "DE": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "DT": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    
    "OLB": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "ILB": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    
    "CB": expand_features(DEFENSIVE_FEATURES + RETURNER_FEATURES + SPECIAL_TEAMS_FEATURE),
    
    "S":  expand_features(DEFENSIVE_FEATURES + RETURNER_FEATURES + SPECIAL_TEAMS_FEATURE),
    
    "K":  expand_features(KICKING_FEATURES + KICK_HOLDING_FEATURE),
    
    "P":  expand_features(PUNTING_FEATURES + KICK_HOLDING_FEATURE),
    
    "LS": expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE + SPECIAL_TEAMS_FEATURE),
}

# Special handling for "All" or if we want to add more general ones
ALL_POSITIONS = list(POSITION_FEATURE_MAP.keys())
