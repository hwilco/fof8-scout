"""Train/inference feature schema contract utilities.

This module defines a serializable schema object used to keep feature
preparation consistent between training and batch inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import polars as pl

FEATURE_SCHEMA_ARTIFACT_PATH = "feature_schema.json"


class FeatureSchemaError(ValueError):
    """Raised when inference features do not match the persisted training schema."""


@dataclass
class FeatureSchema:
    """Schema contract for model feature preparation.

    The schema captures training-time feature column order, categorical
    columns, non-feature exclusions, and mismatch handling policies.
    """

    feature_columns: list[str]
    categorical_columns: list[str]
    excluded_metadata_columns: list[str]
    excluded_target_columns: list[str]
    category_handling_policy: str = "cast_to_string"
    missing_column_policy: str = "reject"
    extra_column_policy: str = "drop"
    missing_defaults: dict[str, Any] = field(default_factory=dict)
    college_bucketing_policy: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def infer_categorical_columns(df: pl.DataFrame) -> list[str]:
        """Infer categorical-like columns from a Polars frame schema."""
        return [
            col for col, dtype in df.schema.items() if dtype in (pl.String, pl.Categorical, pl.Enum)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema for JSON artifact logging."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSchema:
        """Deserialize a schema from an artifact payload."""
        return cls(**payload)

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply schema rules and return model-ready features.

        The returned frame is guaranteed to follow the training column order.
        Missing or extra columns are handled according to schema policies.
        """
        dropped = set(self.excluded_metadata_columns + self.excluded_target_columns)
        candidate_cols = [c for c in df.columns if c not in dropped]
        features_df = df.select(candidate_cols)

        expected = set(self.feature_columns)
        observed = set(features_df.columns)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)

        unresolved_missing = [c for c in missing if c not in self.missing_defaults]
        if unresolved_missing and self.missing_column_policy == "reject":
            raise FeatureSchemaError(
                "Missing required feature columns: "
                f"{unresolved_missing}. "
                "To fix: regenerate features with the training pipeline or provide safe defaults "
                "in the persisted schema."
            )

        fill_exprs = []
        for col in missing:
            if col in self.missing_defaults:
                fill_exprs.append(pl.lit(self.missing_defaults[col]).alias(col))
        if fill_exprs:
            features_df = features_df.with_columns(fill_exprs)

        if extra and self.extra_column_policy == "reject":
            raise FeatureSchemaError(
                "Unexpected extra feature columns: "
                f"{extra}. "
                "To fix: align inference feature generation with training or set extra policy to "
                "'drop' in the persisted schema."
            )
        if extra and self.extra_column_policy == "drop":
            features_df = features_df.drop(extra)

        cat_exprs = []
        if self.category_handling_policy == "cast_to_string":
            for col in self.categorical_columns:
                if col in features_df.columns:
                    cat_exprs.append(pl.col(col).cast(pl.String))
        if cat_exprs:
            features_df = features_df.with_columns(cat_exprs)

        return features_df.select(self.feature_columns)
