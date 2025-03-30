"""
Tools for reshaping data in common ways
"""

from __future__ import annotations

import pandas as pd


def ts_to_long_data(df: pd.DataFrame, time_col_name: str = "time") -> pd.DataFrame:
    """
    Convert timeseries data to long data

    Parameters
    ----------
    df
        Data to convert

    time_col_name
        Name of the time column in the output

    Returns
    -------
    :
        `df` in long-form
    """
    return df.melt(ignore_index=False, var_name=time_col_name).reset_index()
