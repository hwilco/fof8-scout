import polars as pl
from sklearn.preprocessing import StandardScaler


def preprocess_for_sklearn(
    X_pl: pl.DataFrame, scaler=None, expected_columns: list[str] | None = None
):
    """
    Prepares a Polars DataFrame for scikit-learn models by:
    1. One-hot encoding categorical features using pl.to_dummies().
    2. Filling Nulls with 0.
    3. Aligning columns with expected_columns (if provided during inference).
    4. Scaling features using StandardScaler (via numpy conversion).

    Args:
        X_pl: The input Polars DataFrame.
        scaler: An optional pre-fitted StandardScaler object. If None, a new one is fitted.
        expected_columns: Optional list of columns expected after one-hot encoding.

    Returns:
        X_final: Reconstructed Polars DataFrame.
        scaler: The fitted StandardScaler.
        columns: List of columns after one-hot encoding (used as expected_columns later).
    """
    # 1. One-hot encoding
    cat_cols = [
        col for col, dtype in X_pl.schema.items() if dtype in [pl.String, pl.Categorical, pl.Enum]
    ]

    if cat_cols:
        # Use modernized pl.to_dummies
        X_pl = X_pl.to_dummies(columns=cat_cols, drop_first=True)

    # 2. Fill Nulls for GLMs/Linear Models
    X_pl = X_pl.fill_null(0)

    # 3. Align columns if expected_columns are provided (inference time)
    if expected_columns is not None:
        missing_cols = [c for c in expected_columns if c not in X_pl.columns]
        if missing_cols:
            X_pl = X_pl.with_columns([pl.lit(0).alias(c) for c in missing_cols])
        # Select and reorder to match exactly the expected columns
        X_pl = X_pl.select(expected_columns)

    columns = X_pl.columns

    # 4. Scaling (Extracting numpy array)
    X_np = X_pl.to_numpy()

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_np)
    else:
        X_scaled = scaler.transform(X_np)

    # Reconstruct as pure Polars DataFrame before returning
    X_final = pl.DataFrame(X_scaled, schema=columns)

    return X_final, scaler, columns
