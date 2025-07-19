"""
Support for unit conversion
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.index_manipulation import set_index_levels_func
from pandas_openscm.indexing import multi_index_match

if TYPE_CHECKING:
    import pint


def convert_unit_from_target_series(
    df: pd.DataFrame,
    desired_unit: pd.Series[str],
    unit_level: str = "unit",
    ur: pint.UnitRegistry | None = None,
) -> pd.DataFrame:
    missing_rows = df.index.difference(desired_unit.index)
    if not missing_rows.empty:
        msg = "Missing desired unit for {missing_rows}"
        raise AssertionError(msg)

    df_reset_unit = df.reset_index(unit_level)
    df_units = df_reset_unit[unit_level].rename("df_unit")

    # Don't need to align, pandas does that for us.
    # If you want to check, compare the below with
    # unit_map = pd.DataFrame([df_units_s, target_units_s.sample(frac=1)]).T
    unit_map = pd.DataFrame([df_units, desired_unit.rename("target_unit")]).T

    unit_map_no_change = unit_map["df_unit"] == unit_map["target_unit"]
    if unit_map_no_change.all():
        # Already all in desired unit
        return df

    if ur is None:
        try:
            import pint

            ur = pint.get_application_registry()
        except ImportError:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "convert_unit_from_target_series(..., ur=None, ...)", "pint"
            )

    df_no_unit = df_reset_unit.drop(unit_level, axis="columns")
    for (df_unit, target_unit), conversion_df in unit_map[~unit_map_no_change].groupby(
        ["df_unit", "target_unit"]
    ):
        if df_unit == target_unit:
            continue

        to_alter_loc = multi_index_match(df_no_unit.index, conversion_df.index)  # type: ignore
        df_no_unit.loc[to_alter_loc, :] = (
            ur.Quantity(df_no_unit.loc[to_alter_loc, :].values, df_unit)
            .to(target_unit)
            .m
        )

    new_units = (unit_map.reorder_levels(df_no_unit.index.names).loc[df_no_unit.index])[
        "target_unit"
    ]

    res = set_index_levels_func(df_no_unit, {unit_level: new_units}).reorder_levels(
        df.index.names
    )

    return res


def convert_unit(
    df: pd.DataFrame,
    desired_unit: str | Mapping[str, str] | pd.Series[str],
    unit_level: str = "unit",
    ur: pint.UnitRegistry | None = None,
) -> pd.DataFrame:
    """
    Convert a [pd.DataFrame][pandas.DataFrame]'s units

    This uses [convert_unit_from_target_series][].
    If you want to understand the details of how the conversion works,
    see that function's docstring.

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] whose units should be converted

    desired_unit
        Desired unit(s) for `df`

        If this is a string,
        we attempt to convert all timeseries in `df` to the given unit.

        If this is a mapping,
        we convert the given units to the target units.
        Be careful using this form - you need to be certain of the units in `df`.
        If any of your keys don't match the units in `df`
        (even by a single whitespace character)
        then the unit conversion will not happen.

        If this is a [pd.Series][pandas.Series],
        then it will be passed directly to [convert_unit_from_target_series][]
        and the requirements for that function apply.

        For further details, see examples

    unit_level
        Level in `df` which holds unit information

        Passed to [convert_unit_from_target_series][].

    ur
        Unit registry to use for the conversion.

        Passed to [convert_unit_from_target_series][].

    Returns
    -------
    :
        `df` with converted units


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
    df_units_s = df.index.get_level_values(unit_level).to_series(
        index=df.index.droplevel(unit_level), name="df_unit"
    )

    # I don't love creating target_units_s in this function,
    # but it's basically a convenience function
    # and the creation is the only thing that this function does,
    # hence I am ok with it.
    if isinstance(desired_unit, str):
        desired_unit_s = pd.Series(
            [desired_unit] * df.shape[0],
            index=df_units_s.index,
        )

    elif isinstance(desired_unit, Mapping):
        desired_unit_s = df_units_s.replace(desired_unit)

    elif isinstance(desired_unit, pd.Series):
        desired_unit_s = desired_unit

    else:
        raise NotImplementedError(type(desired_unit))

    res = convert_unit_from_target_series(
        df=df, desired_unit=desired_unit_s, unit_level=unit_level, ur=ur
    )

    return res


def convert_unit_like():
    raise NotImplementedError
