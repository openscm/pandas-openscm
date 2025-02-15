"""
Manipulation of the index of data
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

if TYPE_CHECKING:
    P = TypeVar("P", pd.DataFrame, pd.Series[Any])

    import pandas.core.indexes.frozen


def convert_index_to_category_index(pandas_obj: P) -> P:
    """
    Convert the index's values to categories

    This can save a lot of memory and improve the speed of processing.
    However, it comes with some pitfalls.
    For a nice discussion of some of them,
    see [this article](https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a/).

    Parameters
    ----------
    pandas_obj
        Object whose index we want to change to categorical.

    Returns
    -------
    :
        A new object with the same data as `pandas_obj`
        but a category type index.
    """
    new_index = pd.MultiIndex.from_frame(
        pandas_obj.index.to_frame(index=False).astype("category")
    )

    if hasattr(pandas_obj, "columns"):
        return type(pandas_obj)(  # type: ignore # confusing mypy here
            pandas_obj.values,
            index=new_index,
            columns=pandas_obj.columns,
        )

    return type(pandas_obj)(
        pandas_obj.values,
        index=new_index,
    )


def update_index_from_candidates(
    indf: pd.DataFrame, candidates: pandas.core.indexes.frozen.FrozenList
) -> pd.DataFrame:
    """
    Update the index of data to align with the candidate columns as much as possible

    Parameters
    ----------
    indf
        Data of which to update the index

    candidates
        Candidate columns to use to create the updated index

    Returns
    -------
    :
        `indf` with its updated index.

        All columns of `indf` that are in `candidates`
        are used to create the index of the result.

    Notes
    -----
    This overwrites any existing index of `indf`
    so you will only want to use this function
    when you're sure that there isn't anything of interest
    already in the index of `indf`.
    """
    set_to_index = [v for v in candidates if v in indf.columns]
    res = indf.set_index(set_to_index)

    return res
