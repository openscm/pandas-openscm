"""
Manipulation of the index of data
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
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


def unify_index_levels(
    left: pd.MultiIndex, right: pd.MultiIndex
) -> tuple[pd.MultiIndex, pd.MultiIndex]:
    """
    Unify the levels on two indexes

    You can achieve similar behaviour with
    [`pd.DataFrame.align`][pandas.DataFrame.align],
    but we want to do this on indexes
    without paying the price of aligning everything else
    or creating a bunch of NaN that we just drop straight away.

    The levels are unified by simply adding NaN to any level in either `left` or `right`
    that is not in the level of the other index.

    The indexes are returned with the levels from `left` first,
    then the levels from `right`.

    Parameters
    ----------
    left
        First index to unify

    right
        Second index to unify

    Returns
    -------
    left_aligned :
        Left after alignment

    right_aligned :
        Right after alignment

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> idx_a = pd.MultiIndex.from_tuples(
    ...     [
    ...         (1, 2, 3),
    ...         (4, 5, 6),
    ...     ],
    ...     names=["a", "b", "c"],
    ... )
    >>> idx_b = pd.MultiIndex.from_tuples(
    ...     [
    ...         (7, 8),
    ...         (10, 11),
    ...     ],
    ...     names=["a", "b"],
    ... )
    >>> unified_a, unified_b = unify_index_levels(idx_a, idx_b)
    >>> unified_a
    MultiIndex([(1, 2, 3),
                (4, 5, 6)],
               names=['a', 'b', 'c'])

    >>> unified_b
    MultiIndex([( 7,  8, nan),
                (10, 11, nan)],
               names=['a', 'b', 'c'])

    >>> # Also fine if b has swapped levels
    >>> idx_b = pd.MultiIndex.from_tuples(
    ...     [
    ...         (7, 8),
    ...         (10, 11),
    ...     ],
    ...     names=["b", "a"],
    ... )
    >>> unified_a, unified_b = unify_index_levels(idx_a, idx_b)
    >>> unified_a
    MultiIndex([(1, 2, 3),
                (4, 5, 6)],
               names=['a', 'b', 'c'])

    >>> unified_b
    MultiIndex([( 8,  7, nan),
                (11, 10, nan)],
               names=['a', 'b', 'c'])

    >>> # Also works if a is 'inside' b
    >>> idx_a = pd.MultiIndex.from_tuples(
    ...     [
    ...         (7, 8),
    ...         (10, 11),
    ...     ],
    ...     names=["a", "b"],
    ... )
    >>> idx_b = pd.MultiIndex.from_tuples(
    ...     [
    ...         (1, 2, 3),
    ...         (4, 5, 6),
    ...     ],
    ...     names=["a", "b", "c"],
    ... )
    >>> unified_a, unified_b = unify_index_levels(idx_a, idx_b)
    >>> unified_a
    MultiIndex([( 7,  8, nan),
                (10, 11, nan)],
               names=['a', 'b', 'c'])

    >>> unified_b
    MultiIndex([(1, 2, 3),
                (4, 5, 6)],
               names=['a', 'b', 'c'])

    >>> # But, be a bit careful, this is now sensitive to a's column order
    >>> idx_a = pd.MultiIndex.from_tuples(
    ...     [
    ...         (7, 8),
    ...         (10, 11),
    ...     ],
    ...     names=["b", "a"],
    ... )
    >>> idx_b = pd.MultiIndex.from_tuples(
    ...     [
    ...         (1, 2, 3),
    ...         (4, 5, 6),
    ...     ],
    ...     names=["a", "b", "c"],
    ... )
    >>> unified_a, unified_b = unify_index_levels(idx_a, idx_b)
    >>> # Note that the names are `['b', 'a', 'c']` in the output
    >>> unified_a
    MultiIndex([( 7,  8, nan),
                (10, 11, nan)],
               names=['b', 'a', 'c'])

    >>> unified_b
    MultiIndex([(2, 1, 3),
                (5, 4, 6)],
               names=['b', 'a', 'c'])
    """
    if left.names == right.names:
        return left, right

    if (not left.names.difference(right.names)) and (
        not right.names.difference(left.names)
    ):
        return left, right.reorder_levels(left.names)

    joint_idx, left_idxer, right_idxer = left.join(
        right, how="outer", return_indexers=True
    )
    left_aligned = joint_idx[np.where(left_idxer != -1)]
    right_aligned = joint_idx[np.where(right_idxer != -1)]

    return left_aligned, right_aligned
