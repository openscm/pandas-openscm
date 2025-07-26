"""
Unit conversion mix in for accessors
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import pandas as pd

from pandas_openscm.accessors.mix_in_protocol import PandasAccessorLike
from pandas_openscm.unit_conversion import convert_unit, convert_unit_like

if TYPE_CHECKING:
    import pint

    # Unit conversion only works on numeric data
    P = TypeVar("P", pd.DataFrame, pd.Series[float], pd.Series[int])
else:
    # Have to give a value, but doesn't matter what at runtime
    P = TypeVar("P", pd.DataFrame, pd.Series)


class UnitConversionMixIn(Generic[P]):
    """
    Unit conversion mix-in for accessors
    """

    def convert_unit(
        self: PandasAccessorLike[P],
        desired_units: str | Mapping[str, str] | pd.Series[str],
        unit_level: str = "unit",
        ur: pint.facets.PlainRegistry | None = None,
    ) -> P:
        """
        Convert units

        This uses [convert_unit_from_target_series][(p).unit_conversion.].
        If you want to understand the details of how the conversion works,
        see that function's docstring.

        Parameters
        ----------
        desired_units
            Desired unit(s)

            If this is a string,
            we attempt to convert all rows to the given unit.

            If this is a mapping,
            we convert the given units to the target units.
            Be careful using this form - you need to be certain of the units.
            If any of your keys don't match the existing units
            (even by a single whitespace character)
            then the unit conversion will not happen.

            If this is a [pd.Series][pandas.Series],
            then it will be passed to
            [convert_unit_from_target_series][(p).unit_conversion.],
            leaving all rows that are not included unchanged
            (i.e. unspecified rows are not converted).

            For further details, see the examples
            in [convert_unit][(p).unit_conversion.].

        unit_level
            Level in the index which holds unit information

            Passed to [convert_unit_from_target_series][(p).unit_conversion.].

        ur
            Unit registry to use for the conversion.

            Passed to [convert_unit_from_target_series][(p).unit_conversion.].

        Returns
        -------
        :
            Data with converted units
        """
        return convert_unit(
            self.pandas_obj, desired_units=desired_units, unit_level=unit_level, ur=ur
        )

    def convert_unit_like(
        self: PandasAccessorLike[P],
        target: pd.DataFrame | pd.Series[Any],
        unit_level: str = "unit",
        target_unit_level: str | None = None,
        ur: pint.facets.PlainRegistry | None = None,
    ) -> pd.DataFrame:
        """
        Convert units to match another pandas object

        For further details, see the examples
        in [convert_unit_like][(p).unit_conversion.].

        This is essentially a helper for
        [convert_unit_from_target_series][(p).unit_conversion.].
        It implements one set of logic for extracting desired units
        and tries to be clever, handling differences in index levels
        between the data and `target` sensibly wherever possible.

        If you want behaviour other than what is implemented here,
        use [convert_unit_from_target_series][(p).unit_conversion.] directly.

        Parameters
        ----------
        target
            Pandas object whose units should be matched

        unit_level
            Level in the data's index which holds unit information

        target_unit_level
            Level in `target`'s index which holds unit information

            If not supplied, we use `unit_level`.

        ur
            Unit registry to use for the conversion.

            Passed to [convert_unit_from_target_series][(p).unit_conversion.].

        Returns
        -------
        :
            Data with converted units
        """
        return convert_unit_like(
            self.pandas_obj,
            target=target,
            df_unit_level=unit_level,
            target_unit_level=target_unit_level,
            ur=ur,
        )
