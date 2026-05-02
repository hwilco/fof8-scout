import polars as pl
import pytest
from fof8_ml.data.schema import FeatureSchema, FeatureSchemaError


def test_feature_schema_applies_training_column_order():
    schema = FeatureSchema(
        feature_columns=["B", "A", "College"],
        categorical_columns=["College"],
        excluded_metadata_columns=["Player_ID", "Year", "First_Name", "Last_Name"],
        excluded_target_columns=["Cleared_Sieve"],
    )
    frame = pl.DataFrame(
        {
            "Player_ID": [1],
            "Year": [2040],
            "A": [10],
            "College": ["SEC"],
            "B": [20],
            "unused": [999],
        }
    )

    transformed = schema.apply(frame)

    assert transformed.columns == ["B", "A", "College"]
    assert transformed.row(0) == (20, 10, "SEC")


def test_feature_schema_rejects_missing_required_columns():
    schema = FeatureSchema(
        feature_columns=["A", "B"],
        categorical_columns=[],
        excluded_metadata_columns=[],
        excluded_target_columns=[],
        missing_column_policy="reject",
    )

    with pytest.raises(FeatureSchemaError, match="Missing required feature columns"):
        schema.apply(pl.DataFrame({"A": [1]}))


def test_feature_schema_can_fill_missing_defaults():
    schema = FeatureSchema(
        feature_columns=["A", "Age"],
        categorical_columns=[],
        excluded_metadata_columns=[],
        excluded_target_columns=[],
        missing_column_policy="reject",
        missing_defaults={"Age": 22},
    )

    transformed = schema.apply(pl.DataFrame({"A": [7]}))
    assert transformed.columns == ["A", "Age"]
    assert transformed.row(0) == (7, 22)


def test_feature_schema_rejects_extra_columns_when_configured():
    schema = FeatureSchema(
        feature_columns=["A"],
        categorical_columns=[],
        excluded_metadata_columns=[],
        excluded_target_columns=[],
        extra_column_policy="reject",
    )

    with pytest.raises(FeatureSchemaError, match="Unexpected extra feature columns"):
        schema.apply(pl.DataFrame({"A": [1], "B": [2]}))


def test_feature_schema_casts_categoricals_to_string():
    schema = FeatureSchema(
        feature_columns=["Position_Group"],
        categorical_columns=["Position_Group"],
        excluded_metadata_columns=[],
        excluded_target_columns=[],
    )
    frame = pl.DataFrame({"Position_Group": ["OT", "G"]}).with_columns(
        pl.col("Position_Group").cast(pl.Categorical)
    )

    transformed = schema.apply(frame)
    assert transformed.schema["Position_Group"] == pl.String
