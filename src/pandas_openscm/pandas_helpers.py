"""
Helpers for working with pandas

Really these should either go into
[pandas_indexing](https://github.com/coroa/pandas-indexing)
or [pandas](https://github.com/pandas-dev/pandas)
long-term, but they're ok here for now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def multi_index_match(
    idx: pd.MultiIndex, locator: pd.MultiIndex
) -> np.typing.NDArray[np.bool]:
    """
    Perform a multi-index match

    This works, even if the levels of the locator are not the same
    as the levels of the index in which to match.

    Arguably, this should be moved to
    [pandas_indexing](https://github.com/coroa/pandas-indexing)
    or [pandas](https://github.com/pandas-dev/pandas).
    Relevant issues:

    - [pandas#55279](https://github.com/pandas-dev/pandas/issues/55279)
    - [pandas-indexing#64](https://github.com/coroa/pandas-indexing/issues/64)

    Parameters
    ----------
    idx
        Index in which to find matches

    locator
        Locator to use for finding matches

    Returns
    -------
    :
        Location of the rows in `idx` which are in `locator`.

    Examples
    --------
    >>> base = pd.MultiIndex.from_tuples(
    ...     (
    ...         ("ma", "sa", 1),
    ...         ("ma", "sb", 2),
    ...         ("mb", "sa", 1),
    ...         ("mb", "sb", 3),
    ...     ),
    ...     names=["model", "scenario", "id"],
    ... )
    >>>
    >>> # A locator that lines up with the multi-index levels exactly
    >>> loc_simple = pd.MultiIndex.from_tuples(
    ...     (
    ...         ("ma", "sa", 1),
    ...         ("mb", "sa", 1),
    ...     ),
    ...     names=["model", "scenario", "id"],
    ... )
    >>> multi_index_match(base, loc_simple)
    array([ True, False,  True, False])
    >>>
    >>> # A locator that lines up with the first level only
    >>> loc_first_level = pd.MultiIndex.from_tuples(
    ...     (("ma",),),
    ...     names=["model"],
    ... )
    >>> multi_index_match(base, loc_first_level)
    array([ True,  True, False, False])
    >>>
    >>> # A locator that lines up with the second level only
    >>> loc_first_level = pd.MultiIndex.from_tuples(
    ...     (("sa",),),
    ...     names=["scenario"],
    ... )
    >>> multi_index_match(base, loc_first_level)
    array([ True, False,  True, False])
    >>>
    >>> # A locator that lines up with the second and third level only
    >>> loc_first_level = pd.MultiIndex.from_tuples(
    ...     (("sb", 3),),
    ...     names=["scenario", "id"],
    ... )
    >>> multi_index_match(base, loc_first_level)
    array([False, False, False,  True])
    """
    idx_reordered: pd.MultiIndex = idx.reorder_levels(  # type: ignore # reorder_levels untyped
        [*locator.names, *(set(idx.names) - {*locator.names})]
    )

    return idx_reordered.isin(locator)


def multi_index_lookup(df: pd.DataFrame, locator: pd.MultiIndex) -> pd.DataFrame:
    """
    Perform a multi-index look up

    For the problem this is solving, see [`multi_index_match`][(m)].

    Parameters
    ----------
    df
        [`pd.DataFrame`][pandas.DataFrame] in which to find matches

    locator
        Locator to use for finding matches

    Returns
    -------
    :
        Rows of `df` that are in `locator`.

    Examples
    --------
    >>> base = pd.DataFrame(
    ...     data=np.arange(8).reshape((4, 2)),
    ...     columns=[2000, 2020],
    ...     index=pd.MultiIndex.from_tuples(
    ...         (
    ...             ("ma", "sa", 1),
    ...             ("ma", "sb", 2),
    ...             ("mb", "sa", 4),
    ...             ("mb", "sb", 3),
    ...         ),
    ...         names=["model", "scenario", "id"],
    ...     ),
    ... )
    >>>
    >>> # A locator that lines up with the second and third level only
    >>> loc_first_level = pd.MultiIndex.from_tuples(
    ...     (
    ...         ("sa", 1),
    ...         ("sb", 3),
    ...     ),
    ...     names=["scenario", "id"],
    ... )
    >>> multi_index_lookup(base, loc_first_level)
                       2000  2020
    model scenario id
    ma    sa       1      0     1
    mb    sb       3      6     7
    """
    if not isinstance(df.index, pd.MultiIndex):
        msg = (
            "This function is only intended to be used "
            "when `df`'s index is a `MultiIndex`. "
            f"Received {type(df.index)=}"
        )
        raise TypeError(msg)

    return df.loc[multi_index_match(df.index, locator)]
