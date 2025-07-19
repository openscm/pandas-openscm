"""
Support for unit conversion
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.index_manipulation import set_index_levels_func
from pandas_openscm.indexing import multi_index_lookup, multi_index_match

if TYPE_CHECKING:
    import pint


def convert_unit(
    df: pd.DataFrame,
    desired_unit: str | Mapping[str, str],
    unit_level: str = "unit",
    ur: pint.UnitRegistry | None = None,
) -> pd.DataFrame:
    if ur is None:
        try:
            import openscm_units

            ur = openscm_units.unit_registry
        except ImportError:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "convert_unit_like(..., ur=None, ...)", "openscm_units"
            )

    df_units_s = df.index.get_level_values(unit_level).to_series(
        index=df.index.droplevel(unit_level), name="df_unit"
    )

    if isinstance(desired_unit, str):
        unit_map = df_units_s.to_frame()
        unit_map["target_unit"] = desired_unit

    elif isinstance(desired_unit, Mapping):
        # TODO: add check for any keys in desired_unit which are not units in df.
        # Optionally raise if there are extra keys to avoid silent failure.
        target_units_s = df_units_s.map(desired_unit).dropna().rename("target_unit")
        unit_map = pd.DataFrame(
            [df_units_s.loc[target_units_s.index], target_units_s]
        ).T

    elif isinstance(desired_unit, pd.Series):
        # Don't do this,
        # just split out a function which takes in a Series of target_units
        # (and do direct passthrough if the user supplies a Series)
        raise NotImplementedError
        # Assume that desired_unit is already the target units
        unit_map = pd.DataFrame([*desired_unit.align(df_units_s)]).T

    else:
        raise NotImplementedError(type(desired_unit))

    unit_map_no_change = unit_map["df_unit"] == unit_map["target_unit"]
    if unit_map_no_change.all():
        # Already all in desired unit
        return df

    df_no_unit = df.reset_index(unit_level, drop=True)

    for (df_unit, target_unit), conversion_df in unit_map[~unit_map_no_change].groupby(
        ["df_unit", "target_unit"]
    ):
        to_alter_loc = multi_index_match(df_no_unit.index, conversion_df.index)  # type: ignore
        df_no_unit.loc[to_alter_loc, :] = (
            ur.Quantity(df_no_unit.loc[to_alter_loc, :].values, df_unit)
            .to(target_unit)
            .m
        )

    missing_from_unit_map = df_no_unit.index.difference(unit_map.index)
    if not missing_from_unit_map.empty:
        missing_from_unit_map_df = (
            multi_index_lookup(df, missing_from_unit_map)
            .index.to_frame()[[unit_level]]
            .rename({unit_level: "df_unit"}, axis="columns")
        )
        missing_from_unit_map_df["target_unit"] = missing_from_unit_map_df["df_unit"]

        unit_map = pd.concat([unit_map, missing_from_unit_map_df])

    new_units = (unit_map.reorder_levels(df_no_unit.index.names).loc[df_no_unit.index])[
        "target_unit"
    ]

    res = set_index_levels_func(df_no_unit, {unit_level: new_units}).reorder_levels(
        df.index.names
    )

    return res


def convert_unit_like():
    raise NotImplementedError
