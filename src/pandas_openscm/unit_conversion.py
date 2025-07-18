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
        pass

    else:
        raise NotImplementedError(type(desired_unit))

    if (unit_map["df_unit"] == unit_map["target_unit"]).all():
        # Already in matching units
        return df

    df_converted = df.reset_index(unit_level, drop=True)
    for (df_unit, target_unit), conversion_df in unit_map.groupby(
        ["df_unit", "target_unit"]
    ):
        to_alter_loc = multi_index_match(df_converted.index, conversion_df.index)  # type: ignore
        df_converted.loc[to_alter_loc, :] = (
            ur.Quantity(df_converted.loc[to_alter_loc, :].values, df_unit)
            .to(target_unit)
            .m
        )

    # All conversions done so can simply assign the unit column.
    unit_map_reordered = unit_map.reorder_levels(df_converted.index.names)
    res = set_index_levels_func(
        df_converted,
        # use `.loc` to ensure that the values line up with the converted result
        {unit_level: unit_map_reordered["target_unit"].loc[df_converted.index]},
    ).reorder_levels(df.index.names)

    return res


def convert_unit_like():
    raise NotImplementedError
