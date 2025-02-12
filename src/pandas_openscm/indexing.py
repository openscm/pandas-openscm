"""
Helpers for working with pandas

Really these should either go into
[pandas_indexing](https://github.com/coroa/pandas-indexing)
or [pandas](https://github.com/pandas-dev/pandas)
long-term, but they're ok here for now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    P = TypeVar("P", bound=pd.DataFrame | pd.Series[Any])
    import pandas_indexing as pix


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


def multi_index_lookup(pandas_obj: P, locator: pd.MultiIndex) -> pd.DataFrame:
    """
    Perform a multi-index look up

    For the problem this is solving, see [`multi_index_match`][(m)].

    Parameters
    ----------
    pandas_obj
        Pandas object in which to find matches

    locator
        Locator to use for finding matches

    Returns
    -------
    :
        Rows of `pandas_obj` that are in `locator`.

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
    if not isinstance(pandas_obj.index, pd.MultiIndex):
        msg = (
            "This function is only intended to be used "
            "when `df`'s index is a `MultiIndex`. "
            f"Received {type(pandas_obj.index)=}"
        )
        raise TypeError(msg)

    return pandas_obj.loc[multi_index_match(pandas_obj.index, locator)]


def mi_loc(
    pandas_obj: P,
    locator: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector | None = None,
) -> P:
    """
    Select data, being slightly smarter than the default [pandas.DataFrame.loc][].

    Parameters
    ----------
    pandas_obj
        Pandas object on which to do the `.loc` operation

    locator
        Locator to apply

        If this is a multi-index, we use
        [`multi_index_lookup`][(m).] to ensure correct alignment.

        If this is an index that has a name,
        we use the name to ensure correct alignment.

    Returns
    -------
    :
        Selected data
    """
    if isinstance(locator, pd.MultiIndex):
        res = multi_index_lookup(pandas_obj, locator)

    elif isinstance(locator, pd.Index) and locator.name is not None:
        res = pandas_obj[pandas_obj.index.isin(locator.values, level=locator.name)]

    else:
        res = pandas_obj.loc[locator]

    return res
