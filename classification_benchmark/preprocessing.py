from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric values using quantiles learned only from training data."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PercentileClipper":
        """Estimate clipping limits from the training matrix."""

        del y
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned clipping limits."""

        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        """Preserve input feature names after clipping."""

        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


def infer_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical groups."""

    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(
    frame: pd.DataFrame,
    scale_numeric: bool,
    clip_numeric: bool,
) -> ColumnTransformer:
    """Create a leakage-safe preprocessing transformer."""

    numeric_columns, categorical_columns = infer_feature_types(frame)
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if clip_numeric:
        numeric_steps.append(
            ("clipper", PercentileClipper(lower_quantile=0.01, upper_quantile=0.99))
        )
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, object, Sequence[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")
