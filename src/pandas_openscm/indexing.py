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
    P = TypeVar("P", pd.DataFrame, pd.Series[Any])

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

    Raises
    ------
    KeyError
        `locator` has levels which are not in `idx`

    Examples
    --------
    >>> import pandas as pd
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
    try:
        idx_reordered: pd.MultiIndex = idx.reorder_levels(  # type: ignore # reorder_levels untyped
            [*locator.names, *idx.names.difference(locator.names)]  # type: ignore # pandas-stubs confused about difference
        )
    except KeyError as exc:
        unusable = locator.names.difference(idx.names)  # type: ignore # pandas-stubs confused about difference
        if unusable:
            msg = (
                f"The following levels in `locator` are not in `idx`: {unusable}. "
                f"{locator.names=} {idx.names=}"
            )
            raise KeyError(msg) from exc

        raise  # pragma: no cover

    return idx_reordered.isin(locator)


def multi_index_lookup(pandas_obj: P, locator: pd.MultiIndex) -> P:
    """
    Perform a multi-index look up

    For the problem this is solving, see [multi_index_match][(m)].

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
    >>> import numpy as np
    >>> import pandas as pd
    >>>
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
            "when `df`'s index is an instance of `MultiIndex`. "
            f"Received {type(pandas_obj.index)=}"
        )
        raise TypeError(msg)

    res = pandas_obj.loc[multi_index_match(pandas_obj.index, locator)]

    return res


def index_name_aware_match(
    idx: pd.MultiIndex, locator: pd.Index[Any]
) -> np.typing.NDArray[np.bool]:
    """
    Perform a match with an index, being aware of the index's name.

    This works, even if the index being looked up is not the first index.

    Parameters
    ----------
    idx
        Index in which to find matches

    locator
        Locator to use for finding matches

    Returns
    -------
    :
        Location of the rows in `idx` which are in `locator`, given `locator.name`.

    Examples
    --------
    >>> import pandas as pd
    >>>
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
    >>> # A locator that lines up with the third level only
    >>> loc = pd.Index([1, 3], name="id")
    >>> index_name_aware_match(base, loc)
    array([ True, False,  True,  True])
    """
    res = idx.isin(locator.values, level=locator.name)

    return res


def index_name_aware_lookup(pandas_obj: P, locator: pd.Index[Any]) -> P:
    """
    Perform a look up with an index, being aware of the index's name.

    For the problem this is solving, see [index_name_aware_match][(m)].

    Parameters
    ----------
    pandas_obj
        Pandas object in which to find matches

    locator
        Locator to use for finding matches

    Returns
    -------
    :
        Rows of `pandas_obj` that are in `locator`, given `locator.name`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>>
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
    >>> # A locator that lines up with the third level only
    >>> loc = pd.Index([1, 3], name="id")
    >>> index_name_aware_lookup(base, loc)
                       2000  2020
    model scenario id
    ma    sa       1      0     1
    mb    sb       3      6     7
    """
    if not isinstance(pandas_obj.index, pd.MultiIndex):
        msg = (
            "This function is only intended to be used "
            "when `df`'s index is an instance of `MultiIndex`. "
            f"Received {type(pandas_obj.index)=}"
        )
        raise TypeError(msg)

    return pandas_obj.loc[index_name_aware_match(pandas_obj.index, locator)]


def mi_loc(
    pandas_obj: P,
    locator: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector,
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
        [multi_index_lookup][(m).] to ensure correct alignment.

        If this is an index that has a name,
        we use the name to ensure correct alignment.

    Returns
    -------
    :
        Selected data

    Notes
    -----
    If you have [pandas_indexing][] installed,
    you can get the same (perhaps even better) functionality
    using something like the following instead

    ```python
    ...
    pandas_obj.loc[pandas_indexing.isin(locator)]
    ...
    ```
    """
    if isinstance(locator, pd.MultiIndex):
        res: P = multi_index_lookup(pandas_obj, locator)

    elif isinstance(locator, pd.Index) and locator.name is not None:
        res = index_name_aware_lookup(pandas_obj, locator)

    else:
        res = pandas_obj.loc[locator]  # type: ignore

    return res
