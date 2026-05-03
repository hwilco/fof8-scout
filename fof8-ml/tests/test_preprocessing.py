import polars as pl
from fof8_ml.data.preprocessing import preprocess_for_sklearn


def test_preprocess_for_sklearn():
    df = pl.DataFrame(
        {"Feature1": [1, 2, None], "Feature2": [4.0, None, 6.0], "CatFeature": ["A", "B", "A"]}
    )

    X_final, scaler, feature_names = preprocess_for_sklearn(df)

    assert len(feature_names) > 0

    assert X_final.null_count().sum_horizontal().item() == 0
