import pandas as pd
from sklearn.preprocessing import StandardScaler
import polars as pl


def preprocess_for_sklearn(X_pl: pl.DataFrame, scaler=None):
    """
    Prepares a Polars DataFrame for scikit-learn models by:
    1. Converting to pandas.
    2. One-hot encoding categorical features.
    3. Filling NaNs with 0.
    4. Scaling features using StandardScaler.
    """
    X_pd = X_pl.to_pandas()
    # Simple OHE for categorical columns
    cat_cols = X_pd.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_pd = pd.get_dummies(X_pd, columns=cat_cols, drop_first=True)
    # Fill NaNs for GLMs
    X_pd = X_pd.fillna(0)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pd)
    else:
        X_scaled = scaler.transform(X_pd)

    X_final = pd.DataFrame(X_scaled, columns=X_pd.columns, index=X_pd.index)
    return X_final, scaler
