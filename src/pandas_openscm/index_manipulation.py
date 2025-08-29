"""
Manipulation of the index of data
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt
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


def ensure_is_multiindex(index: pd.Index[Any] | pd.MultiIndex) -> pd.MultiIndex:
    """
    Ensure that an index is a [pd.MultiIndex][pandas.MultiIndex]

    Parameters
    ----------
    index
        Index to check

    Returns
    -------
    :
        Index, cast to [pd.MultiIndex][pandas.MultiIndex] if needed
    """
    if isinstance(index, pd.MultiIndex):
        return index

    return pd.MultiIndex.from_arrays([index.values], names=[index.name])


def ensure_index_is_multiindex(pandas_obj: P, copy: bool = True) -> P:
    """
    Ensure that the index of a pandas object is a [pd.MultiIndex][pandas.MultiIndex]

    Parameters
    ----------
    pandas_obj
        Object whose index we want to ensure is a [pd.MultiIndex][pandas.MultiIndex]

    copy
        Should we copy `pandas_obj` before modifying the index?

    Returns
    -------
    :
        `pandas_obj` with a [pd.MultiIndex][pandas.MultiIndex]

        If the index was already a [pd.MultiIndex][pandas.MultiIndex],
        this is a no-op (although the value of copy is respected).
    """
    if copy:
        pandas_obj = pandas_obj.copy()

    if isinstance(pandas_obj.index, pd.MultiIndex):
        return pandas_obj

    pandas_obj.index = ensure_is_multiindex(pandas_obj.index)

    return pandas_obj


def unify_index_levels(
    left: pd.MultiIndex, right: pd.MultiIndex
) -> tuple[pd.MultiIndex, pd.MultiIndex]:
    """
    Unify the levels on two indexes

    The levels are unified by simply adding NaN to any level in either `left` or `right`
    that is not in the level of the other index.

    This is differnt to [pd.DataFrame.align][pandas.DataFrame.align].
    [pd.DataFrame.align][pandas.DataFrame.align]
    will fill missing values with values from the other index if it can.
    We don't want that here.
    We want any non-aligned levels to be filled with NaN.

    The implementation also allows this to be performed on indexes directly
    (avoiding casting to a DataFrame
    and avoiding paying the price of aligning everything else
    or creating a bunch of NaN that we just drop straight away).

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
    >>>
    >>> unified_b
    MultiIndex([( 7,  8, nan),
                (10, 11, nan)],
               names=['a', 'b', 'c'])
    >>>
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
    >>>
    >>> unified_b
    MultiIndex([( 8,  7, nan),
                (11, 10, nan)],
               names=['a', 'b', 'c'])
    >>>
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
    >>>
    >>> unified_b
    MultiIndex([(1, 2, 3),
                (4, 5, 6)],
               names=['a', 'b', 'c'])
    >>>
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
    >>>
    >>> unified_b
    MultiIndex([(2, 1, 3),
                (5, 4, 6)],
               names=['b', 'a', 'c'])
    """
    if left.names == right.names:
        return left, right

    if (not left.names.difference(right.names)) and (  # type: ignore # pandas-stubs confused
        not right.names.difference(left.names)  # type: ignore # pandas-stubs confused
    ):
        return left, right.reorder_levels(left.names)  # type: ignore # pandas-stubs missing reorder_levels

    out_names = [*left.names, *[v for v in right.names if v not in left.names]]
    out_names_s = set(out_names)
    left_to_add = out_names_s.difference(left.names)
    right_to_add = out_names_s.difference(right.names)

    left_unified = pd.MultiIndex(  # type: ignore # pandas-stubs missing reorder_levels
        levels=[
            *left.levels,
            *[np.array([], dtype=right.get_level_values(c).dtype) for c in left_to_add],  # type: ignore # pandas-stubs confused
        ],
        codes=[
            *left.codes,
            *([np.full(left.shape[0], -1)] * len(left_to_add)),
        ],
        names=[
            *left.names,
            *left_to_add,
        ],
    ).reorder_levels(out_names)

    right_unified = pd.MultiIndex(  # type: ignore # pandas-stubs missing reorder_levels
        levels=[
            *[np.array([], dtype=left.get_level_values(c).dtype) for c in right_to_add],  # type: ignore # pandas-stubs confused
            *right.levels,
        ],
        codes=[
            *([np.full(right.shape[0], -1)] * len(right_to_add)),
            *right.codes,
        ],
        names=[
            *right_to_add,
            *right.names,
        ],
    ).reorder_levels(out_names)

    return left_unified, right_unified


def unify_index_levels_check_index_types(
    left: pd.Index[Any], right: pd.Index[Any]
) -> tuple[pd.MultiIndex, pd.MultiIndex]:
    """
    Unify the levels on two indexes

    This is just a thin wrapper around [unify_index_levels][(m).]
    that checks the the inputs are both [pd.MultiIndex][pandas.MultiIndex]
    before unifying the indices.

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
    """
    if not isinstance(left, pd.MultiIndex):
        raise TypeError(left)

    if not isinstance(right, pd.MultiIndex):
        raise TypeError(right)

    return unify_index_levels(left, right)


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


def create_new_level_and_codes_by_mapping(
    ini: pd.MultiIndex,
    level_to_create_from: str,
    mapper: Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any],
) -> tuple[pd.Index[Any], npt.NDArray[np.integer[Any]]]:
    """
    Create a new level and associated codes by mapping an existing level

    This is a thin function intended for internal use
    to handle some slightly tricky logic.

    Parameters
    ----------
    ini
        Input index

    level_to_create_from
        Level to create the new level from

    mapper
        Function to use to map existing levels to new levels

    Returns
    -------
    new_level :
        New level

    new_codes :
        New codes
    """
    level_to_map_from_idx = ini.names.index(level_to_create_from)
    new_level = ini.levels[level_to_map_from_idx].map(mapper)
    if not new_level.has_duplicates:
        # Fast route, can just return new level and codes from level we mapped from
        return new_level, ini.codes[level_to_map_from_idx]

    # Slow route: have to update the codes
    dup_level = ini.get_level_values(level_to_create_from).map(mapper)
    new_level = new_level.unique()
    new_codes = new_level.get_indexer(dup_level)  # type: ignore

    return new_level, new_codes


def create_new_level_and_codes_by_mapping_multiple(
    ini: pd.MultiIndex,
    levels_to_create_from: tuple[str, ...],
    mapper: Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any],
) -> tuple[pd.Index[Any], npt.NDArray[np.integer[Any]]]:
    """
    Create a new level and associated codes by mapping existing levels

    This is a thin function intended for internal use
    to handle some slightly tricky logic.

    Parameters
    ----------
    ini
        Input index

    levels_to_create_from
        Levels to create the new level from

    mapper
        Function to use to map existing levels to new levels

    Returns
    -------
    new_level :
        New level

    new_codes :
        New codes
    """
    # You could probably do some optimisation here
    # that checks for unique combinations of codes
    # for the levels we're using,
    # then only applies the mapping to those unique combos
    # to reduce the number of evaluations of mapper.
    # That feels tricky to get right, so just doing the brute force way for now.
    dup_level = ini.droplevel(
        ini.names.difference(list(levels_to_create_from))  # type: ignore # pandas-stubs confused
    ).map(mapper)

    # Brute force: get codes from new levels
    new_level = dup_level.unique()
    new_codes = new_level.get_indexer(dup_level)

    return new_level, new_codes


def update_index_levels_func(
    pobj: P,
    updates: Mapping[Any, Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any]],
    copy: bool = True,
    remove_unused_levels: bool = True,
) -> P:
    """
    Update the index levels of a [pandas][] object

    Parameters
    ----------
    pobj
        Supported [pandas][] object to update

    updates
        Updates to apply to `pobj`'s index

        Each key is the index level to which the updates will be applied.
        Each value is a function which updates the levels to their new values.

    copy
        Should `pobj` be copied before returning?

    remove_unused_levels
        Call `pobj.index.remove_unused_levels` before updating the levels

        This avoids trying to update levels that aren't being used.

    Returns
    -------
    :
        `pobj` with updates applied to its index
    """
    if copy:
        pobj = pobj.copy()

    if not isinstance(pobj.index, pd.MultiIndex):
        msg = (
            "This function is only intended to be used "
            "when `pobj`'s index is an instance of `MultiIndex`. "
            f"Received {type(pobj.index)=}"
        )
        raise TypeError(msg)

    pobj.index = update_levels(
        pobj.index, updates=updates, remove_unused_levels=remove_unused_levels
    )

    return pobj


def update_levels(
    ini: pd.MultiIndex,
    updates: Mapping[Any, Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any]],
    remove_unused_levels: bool = True,
) -> pd.MultiIndex:
    """
    Update the levels of a [pd.MultiIndex][pandas.MultiIndex]

    Parameters
    ----------
    ini
        Input index

    updates
        Updates to apply

        Each key is the level to which the updates will be applied.
        Each value is a mapper of the form used by
        [pd.Index.map][pandas.Index.map].

    remove_unused_levels
        Call `ini.remove_unused_levels` before updating the levels

        This avoids trying to update levels that aren't being used.

    Returns
    -------
    :
        `ini` with updates applied

    Raises
    ------
    KeyError
        A level in `updates` is not a level in `ini`

    Examples
    --------
    >>> start = pd.MultiIndex.from_tuples(
    ...     [
    ...         ("sa", "ma", "v1", "kg"),
    ...         ("sb", "ma", "v2", "m"),
    ...         ("sa", "mb", "v1", "kg"),
    ...         ("sa", "mb", "v2", "m"),
    ...     ],
    ...     names=["scenario", "model", "variable", "unit"],
    ... )
    >>> start
    MultiIndex([('sa', 'ma', 'v1', 'kg'),
                ('sb', 'ma', 'v2',  'm'),
                ('sa', 'mb', 'v1', 'kg'),
                ('sa', 'mb', 'v2',  'm')],
               names=['scenario', 'model', 'variable', 'unit'])
    >>>
    >>> update_levels(
    ...     start,
    ...     {"model": lambda x: f"model {x}", "scenario": lambda x: f"scenario {x}"},
    ... )
    MultiIndex([('scenario sa', 'model ma', 'v1', 'kg'),
                ('scenario sb', 'model ma', 'v2',  'm'),
                ('scenario sa', 'model mb', 'v1', 'kg'),
                ('scenario sa', 'model mb', 'v2',  'm')],
               names=['scenario', 'model', 'variable', 'unit'])
    >>>
    >>> update_levels(
    ...     start,
    ...     {"variable": {"v1": "variable one", "v2": "variable two"}},
    ... )
    MultiIndex([('sa', 'ma', 'variable one', 'kg'),
                ('sb', 'ma', 'variable two',  'm'),
                ('sa', 'mb', 'variable one', 'kg'),
                ('sa', 'mb', 'variable two',  'm')],
               names=['scenario', 'model', 'variable', 'unit'])
    """
    if remove_unused_levels:
        ini = ini.remove_unused_levels()  # type: ignore

    levels: list[pd.Index[Any]] = list(ini.levels)
    codes: list[npt.NDArray[np.integer[Any]]] = list(ini.codes)

    for level, updater in updates.items():
        if level not in ini.names:
            msg = (
                f"{level} is not available in the index. Available levels: {ini.names}"
            )
            raise KeyError(msg)

        new_level, new_codes = create_new_level_and_codes_by_mapping(
            ini=ini,
            level_to_create_from=level,
            mapper=updater,
        )

        level_idx = ini.names.index(level)
        levels[level_idx] = new_level
        codes[level_idx] = new_codes

    res = pd.MultiIndex(
        levels=levels,
        codes=codes,
        names=ini.names,
    )

    return res


def update_index_levels_from_other_func(
    pobj: P,
    update_sources: dict[
        Any,
        tuple[
            Any,
            Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any],
        ]
        | tuple[
            tuple[Any, ...],
            Callable[[tuple[Any, ...]], Any]
            | dict[tuple[Any, ...], Any]
            | pd.Series[Any],
        ],
    ],
    copy: bool = True,
    remove_unused_levels: bool = True,
) -> P:
    """
    Update the index levels based on other levels of a [pandas][] object

    If the level to be updated doesn't exist,
    it is created.

    Parameters
    ----------
    pobj
        Supported [pandas][] object to update

    update_sources
        Updates to apply to `pobj`'s index

        Each key is the level to which the updates will be applied
        (or the level that will be created if it doesn't already exist).

        There are two options for the values.

        The first is used when only one level is used to update the 'target level'.
        In this case, each value is a tuple of which the first element
        is the level to use to generate the values (the 'source level')
        and the second is mapper of the form used by
        [pd.Index.map][pandas.Index.map]
        which will be applied to the source level
        to update/create the level of interest.

        Each value is a tuple of which the first element
        is the level or levels (if a tuple)
        to use to generate the values (the 'source level')
        and the second is mapper of the form used by
        [pd.Index.map][pandas.Index.map]
        which will be applied to the source level
        to update/create the level of interest.

    copy
        Should `pobj` be copied before returning?

    remove_unused_levels
        Call `pobj.index.remove_unused_levels` before updating the levels

        This avoids trying to update levels that aren't being used.

    Returns
    -------
    :
        `pobj` with updates applied to its index
    """
    if copy:
        pobj = pobj.copy()

    if not isinstance(pobj.index, pd.MultiIndex):
        msg = (
            "This function is only intended to be used "
            "when `pobj`'s index is an instance of `MultiIndex`. "
            f"Received {type(pobj.index)=}"
        )
        raise TypeError(msg)

    pobj.index = update_levels_from_other(
        pobj.index,
        update_sources=update_sources,
        remove_unused_levels=remove_unused_levels,
    )

    return pobj


def update_levels_from_other(
    ini: pd.MultiIndex,
    update_sources: dict[
        Any,
        tuple[
            Any,
            Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any],
        ]
        | tuple[
            tuple[Any, ...],
            Callable[[tuple[Any, ...]], Any]
            | dict[tuple[Any, ...], Any]
            | pd.Series[Any],
        ],
    ],
    remove_unused_levels: bool = True,
) -> pd.MultiIndex:
    """
    Update levels based on other levels in a [pd.MultiIndex][pandas.MultiIndex]

    If the level to be updated doesn't exist,
    it is created.

    Parameters
    ----------
    ini
        Input index

    update_sources
        Updates to apply and their source levels

        Each key is the level to which the updates will be applied
        (or the level that will be created if it doesn't already exist).

        There are two options for the values.

        The first is used when only one level is used to update the 'target level'.
        In this case, each value is a tuple of which the first element
        is the level to use to generate the values (the 'source level')
        and the second is mapper of the form used by
        [pd.Index.map][pandas.Index.map]
        which will be applied to the source level
        to update/create the level of interest.

        Each value is a tuple of which the first element
        is the level or levels (if a tuple)
        to use to generate the values (the 'source level')
        and the second is mapper of the form used by
        [pd.Index.map][pandas.Index.map]
        which will be applied to the source level
        to update/create the level of interest.

    remove_unused_levels
        Call `ini.remove_unused_levels` before updating the levels

        This avoids trying to update based on levels that aren't being used.

    Returns
    -------
    :
        `ini` with updates applied

    Raises
    ------
    KeyError
        A source level in `update_sources` is not a level in `ini`

    Examples
    --------
    >>> start = pd.MultiIndex.from_tuples(
    ...     [
    ...         ("sa", "ma", "v1", "kg"),
    ...         ("sb", "ma", "v2", "m"),
    ...         ("sa", "mb", "v1", "kg"),
    ...         ("sa", "mb", "v2", "m"),
    ...     ],
    ...     names=["scenario", "model", "variable", "unit"],
    ... )
    >>> start
    MultiIndex([('sa', 'ma', 'v1', 'kg'),
                ('sb', 'ma', 'v2',  'm'),
                ('sa', 'mb', 'v1', 'kg'),
                ('sa', 'mb', 'v2',  'm')],
               names=['scenario', 'model', 'variable', 'unit'])
    >>>
    >>> # Create a new level based on an existing level
    >>> update_levels_from_other(
    ...     start,
    ...     {
    ...         "unit squared": ("unit", lambda x: f"{x}**2"),
    ...         "class": ("model", {"ma": "delta", "mb": "gamma"}),
    ...     },
    ... )
    MultiIndex([('sa', 'ma', 'v1', 'kg', 'kg**2', 'delta'),
                ('sb', 'ma', 'v2',  'm',  'm**2', 'delta'),
                ('sa', 'mb', 'v1', 'kg', 'kg**2', 'gamma'),
                ('sa', 'mb', 'v2',  'm',  'm**2', 'gamma')],
               names=['scenario', 'model', 'variable', 'unit', 'unit squared', 'class'])
    >>>
    >>> # Update an existing level based on another level
    >>> update_levels_from_other(
    ...     start,
    ...     {
    ...         "unit": ("variable", {"v1": "g", "v2": "km"}),
    ...         "model": ("scenario", lambda x: f"model {x}"),
    ...     },
    ... )
    MultiIndex([('sa', 'model sa', 'v1',  'g'),
                ('sb', 'model sb', 'v2', 'km'),
                ('sa', 'model sa', 'v1',  'g'),
                ('sa', 'model sa', 'v2', 'km')],
               names=['scenario', 'model', 'variable', 'unit'])
    >>>
    >>> # Create a new level based on multiple existing levels
    >>> update_levels_from_other(
    ...     start,
    ...     {
    ...         "model || scenario": (("model", "scenario"), lambda x: " || ".join(x)),
    ...     },
    ... )
    MultiIndex([('sa', 'ma', 'v1', 'kg', 'sa || ma'),
                ('sb', 'ma', 'v2',  'm', 'sb || ma'),
                ('sa', 'mb', 'v1', 'kg', 'sa || mb'),
                ('sa', 'mb', 'v2',  'm', 'sa || mb')],
               names=['scenario', 'model', 'variable', 'unit', 'model || scenario'])
    >>>
    >>> # Both at the same time
    >>> update_levels_from_other(
    ...     start,
    ...     {
    ...         "title": ("scenario", lambda x: x.capitalize()),
    ...         "unit": ("unit", {"v1": "g", "v2": "km"}),
    ...     },
    ... )
    MultiIndex([('sa', 'ma', 'v1', nan, 'Sa'),
                ('sb', 'ma', 'v2', nan, 'Sb'),
                ('sa', 'mb', 'v1', nan, 'Sa'),
                ('sa', 'mb', 'v2', nan, 'Sa')],
               names=['scenario', 'model', 'variable', 'unit', 'title'])
    >>>
    >>> # Setting with a range of different methods
    >>> update_levels_from_other(
    ...     start,
    ...     {
    ...         # callable
    ...         "y-label": (("variable", "unit"), lambda x: f"{x[0]} ({x[1]})"),
    ...         # dict
    ...         "title": ("scenario", {"sa": "Scenario A", "sb": "Delta"}),
    ...         # pd.Series
    ...         "Source": (
    ...             "model",
    ...             pd.Series(["Internal", "External"], index=["ma", "mb"]),
    ...         ),
    ...     },
    ... )
    MultiIndex([('sa', 'ma', 'v1', 'kg', 'v1 (kg)', 'Scenario A', 'Internal'),
                ('sb', 'ma', 'v2',  'm',  'v2 (m)',      'Delta', 'Internal'),
                ('sa', 'mb', 'v1', 'kg', 'v1 (kg)', 'Scenario A', 'External'),
                ('sa', 'mb', 'v2',  'm',  'v2 (m)', 'Scenario A', 'External')],
               names=['scenario', 'model', 'variable', 'unit', 'y-label', 'title', 'Source'])
    """  # noqa: E501
    if remove_unused_levels:
        ini = ini.remove_unused_levels()  # type: ignore

    levels: list[pd.Index[Any]] = list(ini.levels)
    codes: list[npt.NDArray[np.integer[Any]]] = list(ini.codes)
    names: list[str] = list(ini.names)

    for level, (source, updater) in update_sources.items():
        if isinstance(source, tuple):
            missing_levels = set(source) - set(ini.names)
            if missing_levels:
                conj = "is" if len(missing_levels) == 1 else "are"
                msg = (
                    f"{sorted(missing_levels)} {conj} not available in the index. "
                    f"Available levels: {ini.names}"
                )
                raise KeyError(msg)

            new_level, new_codes = create_new_level_and_codes_by_mapping_multiple(
                ini=ini,
                levels_to_create_from=source,
                mapper=updater,
            )

        else:
            if source not in ini.names:
                msg = (
                    f"{source} is not available in the index. "
                    f"Available levels: {ini.names}"
                )
                raise KeyError(msg)

            new_level, new_codes = create_new_level_and_codes_by_mapping(
                ini=ini,
                level_to_create_from=source,
                mapper=updater,
            )

        if level in ini.names:
            level_idx = ini.names.index(level)
            levels[level_idx] = new_level
            codes[level_idx] = new_codes

        else:
            levels.append(new_level)
            codes.append(new_codes)
            names.append(level)

    res = pd.MultiIndex(levels=levels, codes=codes, names=names)

    return res


def create_level_from_collection(
    level: str, value: Collection[Any]
) -> tuple[pandas.Index[Any], npt.NDArray[np.integer[Any]]]:
    """
    Create new level and corresponding codes.

    Parameters
    ----------
    level
        Name of the level to create

    value
        Values to use to create the level

    Returns
    -------
    :
        New level and corresponding codes
    """
    new_level: pandas.Index[Any] = pd.Index(value, name=level)
    if not new_level.has_duplicates:
        # Fast route, can just return new level and codes from level we mapped from
        return new_level, np.arange(len(value))

    # Slow route, have to update the codes
    new_level = new_level.unique()
    new_codes = new_level.get_indexer(value)  # type: ignore

    return new_level, new_codes


def set_levels(
    ini: pd.MultiIndex, levels_to_set: dict[str, Any | Collection[Any]]
) -> pd.MultiIndex:
    """
    Set the levels of a MultiIndex to the provided values

    Parameters
    ----------
    ini
        Input MultiIndex

    levels_to_set
        Mapping of level names to values to set. If values is of type `Collection`,
        it must be of the same length as the MultiIndex. If it is not a `Collection`,
        it will be set to the same value for all levels.

    Returns
    -------
    :
        New MultiIndex with the levels set to the provided values

    Raises
    ------
    TypeError
        If `ini` is not a MultiIndex
    ValueError
        If the length of the values is a collection that is not equal to the
        length of the index

    Examples
    --------
    >>> start = pd.MultiIndex.from_tuples(
    ...     [
    ...         ("sa", "ma", "v1", "kg"),
    ...         ("sb", "ma", "v2", "m"),
    ...         ("sa", "mb", "v1", "kg"),
    ...         ("sa", "mb", "v2", "m"),
    ...     ],
    ...     names=["scenario", "model", "variable", "unit"],
    ... )
    >>> start
    MultiIndex([('sa', 'ma', 'v1', 'kg'),
                ('sb', 'ma', 'v2',  'm'),
                ('sa', 'mb', 'v1', 'kg'),
                ('sa', 'mb', 'v2',  'm')],
               names=['scenario', 'model', 'variable', 'unit'])
    >>>
    >>> # Set a new level with a single string
    >>> set_levels(
    ...     start,
    ...     {"new_variable": "xyz"},
    ... )
    MultiIndex([('sa', 'ma', 'v1', 'kg', 'xyz'),
            ('sb', 'ma', 'v2',  'm', 'xyz'),
            ('sa', 'mb', 'v1', 'kg', 'xyz'),
            ('sa', 'mb', 'v2',  'm', 'xyz')],
           names=['scenario', 'model', 'variable', 'unit', 'new_variable'])
    >>>
    >>> # Replace a level with a collection
    >>> set_levels(
    ...     start,
    ...     {"new_variable": [1, 2, 3, 4]},
    ... )
    MultiIndex([('sa', 'ma', 'v1', 'kg', 1),
                ('sb', 'ma', 'v2',  'm', 2),
                ('sa', 'mb', 'v1', 'kg', 3),
                ('sa', 'mb', 'v2',  'm', 4)],
               names=['scenario', 'model', 'variable', 'unit', 'new_variable'])
    >>>
    >>> # Replace a level with a single value and add a new level
    >>> set_levels(
    ...     start,
    ...     {"model": "new_model", "new_variable": ["xyz", "xyz", "x", "y"]},
    ... )
    MultiIndex([('sa', 'new_model', 'v1', 'kg', 'xyz'),
                ('sb', 'new_model', 'v2',  'm', 'xyz'),
                ('sa', 'new_model', 'v1', 'kg',   'x'),
                ('sa', 'new_model', 'v2',  'm',   'y')],
               names=['scenario', 'model', 'variable', 'unit', 'new_variable'])
    """
    levels: list[pd.Index[Any]] = list(ini.levels)
    codes: list[npt.NDArray[np.integer[Any]]] = list(ini.codes)
    names: list[str] = list(ini.names)

    for level, value in levels_to_set.items():
        if isinstance(value, Collection) and not isinstance(value, str):
            if len(value) != len(ini):
                msg = (
                    f"Length of values for level '{level}' does not "
                    f"match index length: {len(value)} != {len(ini)}"
                )
                raise ValueError(msg)
            new_level, new_codes = create_level_from_collection(level, value)
        else:
            new_level = pd.Index([value], name=level)
            new_codes = np.zeros(ini.shape[0], dtype=int)

        if level in ini.names:
            level_idx = ini.names.index(level)
            levels[level_idx] = new_level
            codes[level_idx] = new_codes
        else:
            levels.append(new_level)
            codes.append(new_codes)
            names.append(level)

    res = pd.MultiIndex(levels=levels, codes=codes, names=names)

    return res


def set_index_levels_func(
    pobj: P,
    levels_to_set: dict[str, Any | Collection[Any]],
    copy: bool = True,
) -> P:
    """
    Set the index levels of a [pd.DataFrame][pandas.DataFrame]

    Parameters
    ----------
    pobj
        Supported [pandas][] object to update

    levels_to_set
        Mapping of level names to values to set

    copy
        Should `pobj` be copied before returning?

    Returns
    -------
    :
        `pobj` with updates applied to its index
    """
    if not isinstance(pobj.index, pd.MultiIndex):
        msg = (
            "This function is only intended to be used "
            "when `pobj`'s index is an instance of `MultiIndex`. "
            f"Received {type(pobj.index)=}"
        )
        raise TypeError(msg)

    if copy:
        pobj = pobj.copy()

    pobj.index = set_levels(pobj.index, levels_to_set=levels_to_set)  # type: ignore

    return pobj
