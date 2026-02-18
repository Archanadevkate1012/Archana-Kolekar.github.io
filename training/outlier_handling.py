"""Utilities for handling outliers in tabular training data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def remove_outliers_iqr(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Remove rows with outlier values using the 1.5*IQR rule.

    Args:
        df: Input dataframe.
        columns: Numeric columns to evaluate.

    Returns:
        Filtered dataframe with index reset.
    """
    filtered_df = df.copy()

    for col in columns:
        q1 = filtered_df[col].quantile(0.25)
        q3 = filtered_df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]

    return filtered_df.reset_index(drop=True)
