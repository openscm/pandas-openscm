"""
Support for unit conversion
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.index_manipulation import (
    ensure_index_is_multiindex,
    ensure_is_multiindex,
    set_index_levels_func,
)
from pandas_openscm.indexing import multi_index_lookup, multi_index_match

if TYPE_CHECKING:
    P = TypeVar("P", pd.Series[float], pd.Series[int], pd.DataFrame)

    import pint.facets


class MissingDesiredUnitError(ValueError):
    """
    Raised when the desired unit is not specified for all timeseries
    """

    def __init__(self, missing_ts: pd.MultiIndex) -> None:
        """
        Initialise the error

        Parameters
        ----------
        missing_ts
            Timeseries for which no desired unit is specified
        """
        msg = f"Missing desired unit for the following timeseries {missing_ts}"
        super().__init__(msg)


def convert_unit_from_target_series(
    pobj: P,
    desired_units: pd.Series[str],
    unit_level: str = "unit",
    ur: pint.facets.PlainRegistry | None = None,
) -> P:
    """
    Convert `pobj`'s units based on a [pd.Series][pandas.Series]

    `desired_uni` defines the units to convert to.
    This is a relatively low-level function,
    you may find [convert_unit][(m).] and [convert_unit_like][(m).] easier to use.

    Parameters
    ----------
    pobj
        Supported [pandas][] object whose units should be converted

    desired_units
        Desired unit(s) for `pobj`

        This must be a [pd.Series][pandas.Series]
        with an index that contains all the rows in `pobj`.

    unit_level
        Level in `pobj`'s index which holds unit information

    ur
        Unit registry to use for the conversion.

        If not supplied, we use [pint.get_application_registry][].

    Returns
    -------
    :
        `pobj` with converted units

    Raises
    ------
    AssertionError
        `desired_units`'s index does not contain all the rows in `pobj`

    MissingOptionalDependencyError
        `ur` is `None` and [pint](https://pint.readthedocs.io/) is not available.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> start = pd.DataFrame(
    ...     [[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [37.0, 38.1, 37.9]],
    ...     columns=[2020, 2030, 2050],
    ...     index=pd.MultiIndex.from_tuples(
    ...         (
    ...             ("sa", "temperature", "mK"),
    ...             ("sb", "temperature", "K"),
    ...             ("sb", "body temperature", "degC"),
    ...         ),
    ...         names=["scenario", "variable", "unit"],
    ...     ),
    ... )
    >>>
    >>> convert_unit_from_target_series(
    ...     start,
    ...     desired_units=pd.Series(
    ...         ["K", "mK", "degF"],
    ...         index=pd.MultiIndex.from_tuples(
    ...             (
    ...                 ("sa", "temperature"),
    ...                 ("sb", "temperature"),
    ...                 ("sb", "body temperature"),
    ...             ),
    ...             names=["scenario", "variable"],
    ...         ),
    ...     ),
    ... )
                                        2020      2030      2050
    scenario variable         unit
    sa       temperature      K        0.001     0.002     0.003
    sb       temperature      mK    1100.000  1200.000  1300.000
             body temperature degF    98.600   100.580   100.220
    """
    desired_units = ensure_index_is_multiindex(desired_units)

    pobj_rows_checker = ensure_is_multiindex(pobj.index.droplevel(unit_level))
    missing_rows = pobj_rows_checker.difference(  # type: ignore # pandas-stubs missing API
        desired_units.index.reorder_levels(pobj_rows_checker.names)  # type: ignore # pandas-stubs missing API
    )
    if not missing_rows.empty:
        raise MissingDesiredUnitError(missing_rows)

    pobj_units = pd.Series(
        pobj.index.get_level_values(unit_level),
        index=ensure_is_multiindex(pobj.index.droplevel(unit_level)),
    )

    desired_units_in_pobj = multi_index_lookup(desired_units, pobj_units.index)  # type: ignore # already checked that pobj.index is MultiIndex

    # Don't need to align, pandas does that for us.
    # If you want to check, compare the below with
    # unit_map = pd.DataFrame([pobj_units, desired_units_in_pobj.sample(frac=1)]).T
    unit_map = pd.DataFrame(
        [pobj_units.rename("pobj_unit"), desired_units_in_pobj.rename("target_unit")]
    ).T
    unit_changes = unit_map["pobj_unit"] != unit_map["target_unit"]
    if not unit_changes.any():
        # Already all in desired unit
        return pobj

    if ur is None:
        try:
            import pint

            ur = pint.get_application_registry()  # type: ignore # pint typing limited
        except ImportError:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "convert_unit_from_target_series(..., ur=None, ...)", "pint"
            )

    pobj_no_unit = ensure_index_is_multiindex(pobj.reset_index(unit_level, drop=True))
    for (pobj_unit, target_unit), conversion_df in unit_map[unit_changes].groupby(
        ["pobj_unit", "target_unit"]
    ):
        to_alter_loc = multi_index_match(pobj_no_unit.index, conversion_df.index)  # type: ignore
        pobj_no_unit.loc[to_alter_loc, :] = (
            ur.Quantity(pobj_no_unit.loc[to_alter_loc, :].values, pobj_unit)
            .to(target_unit)
            .m
        )

    new_units = (
        unit_map.reorder_levels(pobj_no_unit.index.names).loc[pobj_no_unit.index]
    )["target_unit"]
    res = set_index_levels_func(pobj_no_unit, {unit_level: new_units}).reorder_levels(
        pobj.index.names
    )

    return res


def convert_unit(
    pobj: P,
    desired_units: str | Mapping[str, str] | pd.Series[str],
    unit_level: str = "unit",
    ur: pint.facets.PlainRegistry | None = None,
) -> P:
    """
    Convert a supported [pandas][] object's units

    This uses [convert_unit_from_target_series][(m).].
    If you want to understand the details of how the conversion works,
    see that function's docstring.

    Parameters
    ----------
    pobj
        Supported [pandas][] object whose units should be converted

    desired_units
        Desired unit(s) for `pobj`

        If this is a string,
        we attempt to convert all timeseries in `pobj` to the given unit.

        If this is a mapping,
        we convert the given units to the target units.
        Be careful using this form - you need to be certain of the units in `pobj`.
        If any of your keys don't match the units in `pobj`
        (even by a single whitespace character)
        then the unit conversion will not happen.

        If this is a [pd.Series][pandas.Series],
        then it will be passed to [convert_unit_from_target_series][(m).]
        after filling any rows in `pobj` that are not in `desired_units`
        with the unit from `pobj` (i.e. unspecified rows are not converted).

        For further details, see examples

    unit_level
        Level in `pobj`'s index which holds unit information

        Passed to [convert_unit_from_target_series][(m).].

    ur
        Unit registry to use for the conversion.

        Passed to [convert_unit_from_target_series][(m).].

    Returns
    -------
    :
        `pobj` with converted units

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> start = pd.DataFrame(
    ...     [[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [37.0, 38.1, 37.9]],
    ...     columns=[2020, 2030, 2050],
    ...     index=pd.MultiIndex.from_tuples(
    ...         (
    ...             ("sa", "temperature", "mK"),
    ...             ("sb", "temperature", "K"),
    ...             ("sb", "body temperature", "degC"),
    ...         ),
    ...         names=["scenario", "variable", "unit"],
    ...     ),
    ... )
    >>>
    >>> # Convert all timeseries to a given unit
    >>> convert_unit(start, "K")
                                       2020     2030     2050
    scenario variable         unit
    sa       temperature      K       0.001    0.002    0.003
    sb       temperature      K       1.100    1.200    1.300
             body temperature K     310.150  311.250  311.050
    >>>
    >>> # Same thing with a series as input
    >>> convert_unit(start[2030], "K")
    scenario  variable          unit
    sa        temperature       K         0.002
    sb        temperature       K         1.200
              body temperature  K       311.250
    Name: 2030, dtype: float64
    >>>
    >>> # Convert using a mapping.
    >>> # Units that aren't specified in the mapping aren't converted.
    >>> convert_unit(start, {"mK": "K", "K": "kK"})
                                       2020     2030     2050
    scenario variable         unit
    sa       temperature      K      0.0010   0.0020   0.0030
    sb       temperature      kK     0.0011   0.0012   0.0013
             body temperature degC  37.0000  38.1000  37.9000
    >>>
    >>> # When using a mapping, be careful.
    >>> # If you have a typo, there will be no conversion but also no error.
    >>> convert_unit(start, {"MK": "K", "K": "kK"})
                                       2020     2030     2050
    scenario variable         unit
    sa       temperature      mK     1.0000   2.0000   3.0000
    sb       temperature      kK     0.0011   0.0012   0.0013
             body temperature degC  37.0000  38.1000  37.9000
    >>>
    >>> # Convert using a series
    >>> convert_unit(
    ...     start,
    ...     pd.Series(
    ...         ["K", "degF"],
    ...         index=pd.MultiIndex.from_tuples(
    ...             (
    ...                 ("sa", "temperature"),
    ...                 # ("sb", "temperature") not included therefore not converted
    ...                 ("sb", "body temperature"),
    ...             ),
    ...             names=["scenario", "variable"],
    ...         ),
    ...     ),
    ... )
                                      2020     2030     2050
    scenario variable         unit
    sa       temperature      K      0.001    0.002    0.003
    sb       temperature      K      1.100    1.200    1.300
             body temperature degF  98.600  100.580  100.220
    """
    pobj_units_s = ensure_index_is_multiindex(
        pobj.index.get_level_values(unit_level).to_series(
            index=pobj.index.droplevel(unit_level), name="pobj_unit"
        )
    )

    # I don't love creating target_units_s in this function,
    # but it's basically a convenience function
    # and the creation is the only thing that this function does,
    # hence I am ok with it.
    if isinstance(desired_units, str):
        desired_units_s = pd.Series(
            [desired_units] * pobj.shape[0],
            index=pobj_units_s.index,
        )

    elif isinstance(desired_units, Mapping):
        desired_units_s = pobj_units_s.replace(desired_units)  # type: ignore # pandas-stubs missing Mapping option

    elif isinstance(desired_units, pd.Series):
        desired_units = ensure_index_is_multiindex(desired_units)

        missing = pobj_units_s.index.difference(desired_units.index)
        if missing.empty:
            desired_units_s = desired_units
        else:
            desired_units_s = pd.concat(
                [desired_units, multi_index_lookup(pobj_units_s, missing)]
            )

    else:
        raise NotImplementedError(type(desired_units))

    res = convert_unit_from_target_series(
        pobj=pobj, desired_units=desired_units_s, unit_level=unit_level, ur=ur
    )

    return res


class AmbiguousTargetUnitError(ValueError):
    """
    Raised when `target` provided to `convert_unit_like` gives ambiguous desired units
    """

    def __init__(self, msg: str) -> None:
        """
        Initialise the error

        Parameters
        ----------
        msg
            Message to provide to the user
        """
        super().__init__(msg)


def convert_unit_like(
    pobj: P,
    target: pd.DataFrame | pd.Series[Any],
    unit_level: str = "unit",
    target_unit_level: str | None = None,
    ur: pint.facets.PlainRegistry | None = None,
) -> P:
    """
    Convert units to match another [pd.DataFrame][pandas.DataFrame]

    This is essentially a helper function for [convert_unit_from_target_series][(m).].
    It implements one set of logic for extracting desired units and tries to be clever,
    handling differences in index levels
    between `pobj` and `target` sensibly wherever possible.

    If you want behaviour other than what is implemented here,
    use [convert_unit_from_target_series][(m).] directly.

    Parameters
    ----------
    pobj
        Supported [pandas][] object whose units should be converted

    target
        Supported [pandas][] object whose units should be matched

    unit_level
        Level in `pobj`'s index which holds unit information

    target_unit_level
        Level in `target`'s index which holds unit information

        If not supplied, we use `unit_level`.

    ur
        Unit registry to use for the conversion.

        Passed to [convert_unit_from_target_series][(m).].

    Returns
    -------
    :
        `pobj` with converted units

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> start = pd.DataFrame(
    ...     [
    ...         [1010.0, 2010.0, 1150.0],
    ...         [100.1, 100.3, 99.8],
    ...         [0.0011, 0.0012, 0.0013],
    ...         [310_000, 311_000, 310_298],
    ...     ],
    ...     columns=[2020, 2030, 2050],
    ...     index=pd.MultiIndex.from_tuples(
    ...         (
    ...             ("sa", "temperature", "mK"),
    ...             ("sa", "body temperature", "degF"),
    ...             ("sb", "temperature", "kK"),
    ...             ("sb", "body temperature", "mK"),
    ...         ),
    ...         names=["scenario", "variable", "unit"],
    ...     ),
    ... )
    >>>
    >>> target = pd.DataFrame(
    ...     [[1.0, 2.0], [1.1, 1.2]],
    ...     columns=[1990.0, 2010.0],
    ...     index=pd.MultiIndex.from_tuples(
    ...         (
    ...             ("temperature", "K"),
    ...             ("body temperature", "degC"),
    ...         ),
    ...         names=["variable", "unit"],
    ...     ),
    ... )
    >>>
    >>> convert_unit_like(start, target)
                                         2020       2030       2050
    scenario variable         unit
    sa       temperature      K      1.010000   2.010000   1.150000
             body temperature degC  37.833333  37.944444  37.666667
    sb       temperature      K      1.100000   1.200000   1.300000
             body temperature degC  36.850000  37.850000  37.148000
    """
    if target_unit_level is None:
        target_unit_level_use = unit_level
    else:
        target_unit_level_use = target_unit_level

    pobj_units_s = ensure_index_is_multiindex(
        pobj.index.get_level_values(unit_level).to_series(
            index=pobj.index.droplevel(unit_level)
        )
    )

    extra_index_levels_target = target.index.names.difference(  # type: ignore # pandas-stubs API out of date
        [*pobj.index.names, target_unit_level_use]
    )
    if extra_index_levels_target:
        # Drop out the extra levels and duplicates,
        # then create the target units Series
        # (ambiguity in the result is handled later)
        target_index_without_extra_levels_and_dups = target.index.droplevel(
            extra_index_levels_target
        ).drop_duplicates()
        target_units_s = target_index_without_extra_levels_and_dups.get_level_values(
            target_unit_level_use
        ).to_series(
            index=target_index_without_extra_levels_and_dups.droplevel(
                target_unit_level_use
            )
        )

    else:
        target_units_s = target.index.get_level_values(target_unit_level_use).to_series(
            index=target.index.droplevel(target_unit_level_use)
        )

    target_units_s = ensure_index_is_multiindex(target_units_s)

    ambiguous = target_units_s.index.duplicated(keep=False)
    if ambiguous.any():
        ambiguous_idx = target_units_s[ambiguous].index.remove_unused_levels()
        if not isinstance(target.index, pd.MultiIndex):  # pragma: no cover
            # Should be unreachable, but just in case
            raise TypeError(type(target.index))

        ambiguous_drivers = target.index[multi_index_match(target.index, ambiguous_idx)]

        msg = (
            f"`pobj` has {pobj.index.names=}. "
            f"`target` has {target.index.names=}. "
            "The index levels in `target` that are also in `pobj` are "
            f"{target_units_s.index.names}. "
            "When we only look at these levels, the desired unit looks like:\n"
            f"{target_units_s}\n"
            "The unit to use isn't unambiguous for the following metadata:\n"
            f"{target_units_s[ambiguous]}\n"
            "The drivers of this ambiguity "
            "are the following metadata levels in `target`\n"
            f"{ambiguous_drivers}"
        )
        raise AmbiguousTargetUnitError(msg)

    target_units_s, _ = target_units_s.align(pobj_units_s)
    target_units_s = target_units_s.reorder_levels(pobj_units_s.index.names)
    if target_units_s.isnull().any():
        # Fill rows that don't get a spec with their existing units
        target_units_s = multi_index_lookup(
            target_units_s,
            pobj_units_s.index,  # type: ignore # checked that index is MultiIndex above
        ).fillna(pobj_units_s)

    res = convert_unit_from_target_series(
        pobj=pobj, desired_units=target_units_s, unit_level=unit_level, ur=ur
    )

    return res
