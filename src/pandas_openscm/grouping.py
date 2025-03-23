"""
Support for grouping in various ways
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def groupby_except(
    df: pd.DataFrame, non_groupers: str | list[str], observed: bool = True
) -> pd.core.groupby.generic.DataFrameGroupBy[Any]:
    """
    Group by all index levels except specified levels

    This is the inverse of [pd.DataFrame.groupby][pandas.DataFrame.groupby].

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to group

    non_groupers
        Columns to exclude from the grouping

    observed
        Whether to only return observed combinations or not

    Returns
    -------
    :
        `df`, grouped by all columns except `non_groupers`.
    """
    if isinstance(non_groupers, str):
        non_groupers = [non_groupers]

    return df.groupby(df.index.names.difference(non_groupers), observed=observed)  # type: ignore # pandas-stubs confused


def fix_index_name_after_groupby_quantile(
    df: pd.DataFrame, new_name: str = "quantile", copy: bool = False
) -> pd.DataFrame:
    """
    Fix the index name after performing a `groupby(...).quantile(...)` operation

    By default, pandas doesn't assign a name to the quantile level
    when doing an operation of the form given above.
    This fixes this, but it does assume
    that the quantile level is the only unnamed level in the index.

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] of which to fix the name

    new_name
        New name to give to the quantile column

    copy
        Whether to copy `df` before manipulating the index name

    Returns
    -------
    :
        `df`, with the last level in its index renamed to `new_name`.
    """
    if copy:
        res = df.copy()
    else:
        res = df

    res.index = res.index.rename({None: new_name})

    return res
