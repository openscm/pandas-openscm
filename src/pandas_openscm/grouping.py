"""
Support for grouping in various ways
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import pandas as pd

if TYPE_CHECKING:
    P = TypeVar("P", pd.DataFrame, pd.Series[Any])


@overload
def groupby_except(
    pandas_obj: pd.DataFrame, non_groupers: str | list[str], observed: bool = True
) -> pd.core.groupby.generic.DataFrameGroupBy[Any]: ...


@overload
def groupby_except(
    pandas_obj: pd.Series[Any], non_groupers: str | list[str], observed: bool = True
) -> pd.core.groupby.generic.SeriesGroupBy[Any, Any]: ...


def groupby_except(
    pandas_obj: pd.DataFrame | pd.Series[Any],
    non_groupers: str | list[str],
    observed: bool = True,
) -> (
    pd.core.groupby.generic.DataFrameGroupBy[Any]
    | pd.core.groupby.generic.SeriesGroupBy[Any, Any]
):
    """
    Group by all index levels except specified levels

    This is the inverse of [pd.DataFrame.groupby][pandas.DataFrame.groupby].

    Parameters
    ----------
    pandas_obj
        Object to group

    non_groupers
        Columns to exclude from the grouping

    observed
        Whether to only return observed combinations or not

    Returns
    -------
    :
        Object, grouped by all columns except `non_groupers`.
    """
    if isinstance(non_groupers, str):
        non_groupers = [non_groupers]

    return pandas_obj.groupby(
        pandas_obj.index.names.difference(non_groupers),  # type: ignore # pandas-stubs confused
        observed=observed,
    )


def fix_index_name_after_groupby_quantile(
    pandas_obj: P, new_name: str = "quantile", copy: bool = False
) -> P:
    """
    Fix the index name after performing a `groupby(...).quantile(...)` operation

    By default, pandas doesn't assign a name to the quantile level
    when doing an operation of the form given above.
    This fixes this, but it does assume
    that the quantile level is the only unnamed level in the index.

    Parameters
    ----------
    pandas_obj
        Object of which we want to fix the name

    new_name
        New name to give to the quantile column

    copy
        Whether to copy the object before manipulating the index name

    Returns
    -------
    :
        Object, with the last level in its index renamed to `new_name`.
    """
    if copy:
        res = pandas_obj.copy()
    else:
        res = pandas_obj

    res.index = res.index.rename({None: new_name})

    return res
