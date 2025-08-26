"""
Accessor for [pd.Series][pandas.Series]
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import pandas as pd

from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import (
    ensure_index_is_multiindex,
    set_index_levels_func,
)
from pandas_openscm.indexing import mi_loc
from pandas_openscm.unit_conversion import convert_unit, convert_unit_like

if TYPE_CHECKING:
    # Hmm this is somehow not correct.
    # Figuring it out is a job for another day
    S = TypeVar("S", bound=pd.Series[Any])

    import pandas_indexing as pix
    import pint

else:
    S = TypeVar("S")


class PandasSeriesOpenSCMAccessor(Generic[S]):
    """
    [pd.Series][pandas.Series] accessor

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, series: S):
        """
        Initialise

        Parameters
        ----------
        series
            [pd.Series][pandas.Series] to use via the accessor
        """
        # It is possible to validate here.
        # However, it's probably better to do validation closer to the data use.
        self._series = series

    def convert_unit(
        self,
        desired_units: str | Mapping[str, str] | pd.Series[str],
        unit_level: str = "unit",
        ur: pint.facets.PlainRegistry | None = None,
    ) -> S:
        """
        Convert units

        This uses [convert_unit_from_target_series][pandas_openscm.unit_conversion.].
        If you want to understand the details of how the conversion works,
        see that function's docstring.

        Parameters
        ----------
        desired_units
            Desired unit(s) for `series`

            If this is a string,
            we attempt to convert all timeseries to the given unit.

            If this is a mapping,
            we convert the given units to the target units.
            Be careful using this form - you need to be certain of the units.
            If any of your keys don't match the existing units
            (even by a single whitespace character)
            then the unit conversion will not happen.

            If this is a [pd.Series][pandas.Series],
            then it will be passed to
            [convert_unit_from_target_series][pandas_openscm.unit_conversion.]
            after filling any rows in the [pd.Series][pandas.Series]
            that are not in `desired_units`
            with the existing unit (i.e. unspecified rows are not converted).

            For further details, see the examples
            in [convert_unit][pandas_openscm.unit_conversion.].

        unit_level
            Level in the index which holds unit information

            Passed to
            [convert_unit_from_target_series][pandas_openscm.unit_conversion.].

        ur
            Unit registry to use for the conversion.

            Passed to
            [convert_unit_from_target_series][pandas_openscm.unit_conversion.].

        Returns
        -------
        :
            Data with converted units
        """
        res = convert_unit(
            self._series, desired_units=desired_units, unit_level=unit_level, ur=ur
        )

        # The type hinting is impossible to get right here
        # because the casting doesn't work to match the return type
        # (the return type is the same as the input,
        # but we would have to cast to make sure it's numeric
        # and we can't do a runtime check because pd.Series
        # is not subscriptable at runtime).
        # Hence just ignore the type stuff,
        # it's impossible to get right with pandas' accessor pattern.
        # If users want correct type hints, they should use the functional form.
        return res  # type: ignore

    def convert_unit_like(
        self,
        target: pd.DataFrame | pd.Series[Any],
        unit_level: str = "unit",
        target_unit_level: str | None = None,
        ur: pint.facets.PlainRegistry | None = None,
    ) -> S:
        """
        Convert units to match another supported pandas object

        For further details, see the examples
        in [convert_unit_like][pandas_openscm.unit_conversion.].

        This is essentially a helper for
        [convert_unit_from_target_series][pandas_openscm.unit_conversion.].
        It implements one set of logic for extracting desired units
        and tries to be clever, handling differences in index levels
        between the data and `target` sensibly wherever possible.

        If you want behaviour other than what is implemented here,
        use [convert_unit_from_target_series][pandas_openscm.unit_conversion.] directly.

        Parameters
        ----------
        target
            Supported [pandas][] object whose units should be matched

        unit_level
            Level in the data's index which holds unit information

        target_unit_level
            Level in `target`'s index which holds unit information

            If not supplied, we use `unit_level`.

        ur
            Unit registry to use for the conversion.

            Passed to
            [convert_unit_from_target_series][pandas_openscm.unit_conversion.].

        Returns
        -------
        :
            Data with converted units
        """
        res = convert_unit_like(
            self._series,
            target=target,
            unit_level=unit_level,
            target_unit_level=target_unit_level,
            ur=ur,
        )

        # The type hinting is impossible to get right here
        # because the casting doesn't work to match the return type
        # (the return type is the same as the input,
        # but we would have to cast to make sure it's numeric
        # and we can't do a runtime check because pd.Series
        # is not subscriptable at runtime).
        # Hence just ignore the type stuff,
        # it's impossible to get right with pandas' accessor pattern.
        # If users want correct type hints, they should use the functional form.
        return res  # type: ignore

    def ensure_index_is_multiindex(self, copy: bool = True) -> S:
        """
        Ensure that the index is a [pd.MultiIndex][pandas.MultiIndex]

        Parameters
        ----------
        copy
            Whether to copy `series` before manipulating the index name

        Returns
        -------
        :
            `series` with a [pd.MultiIndex][pandas.MultiIndex]

            If the index was already a [pd.MultiIndex][pandas.MultiIndex],
            this is a no-op (although the value of copy is respected).
        """
        res = ensure_index_is_multiindex(self._series, copy=copy)

        return res  # type: ignore # something wront with generic type hinting

    def eiim(self, copy: bool = True) -> S:
        """
        Ensure that the index is a [pd.MultiIndex][pandas.MultiIndex]

        Alias for [ensure_index_is_multiindex][pandas_openscm.index_manipulation.]

        Parameters
        ----------
        copy
            Whether to copy `series` before manipulating the index name

        Returns
        -------
        :
            `series` with a [pd.MultiIndex][pandas.MultiIndex]

            If the index was already a [pd.MultiIndex][pandas.MultiIndex],
            this is a no-op (although the value of copy is respected).
        """
        return self.ensure_index_is_multiindex(copy=copy)

    def fix_index_name_after_groupby_quantile(
        self, new_name: str = "quantile", copy: bool = False
    ) -> S:
        """
        Fix the index name after performing a `groupby(...).quantile(...)` operation

        By default, pandas doesn't assign a name to the quantile level
        when doing an operation of the form given above.
        This fixes this, but it does assume
        that the quantile level is the only unnamed level in the index.

        Parameters
        ----------
        new_name
            New name to give to the quantile column

        copy
            Whether to copy `series` before manipulating the index name

        Returns
        -------
        :
            `series`, with the last level in its index renamed to `new_name`.
        """
        res = fix_index_name_after_groupby_quantile(
            self._series, new_name=new_name, copy=copy
        )

        # Ignore return type
        # because I've done something wrong with how I've set this up.
        # Figuring this out is a job for another day
        return res  # type: ignore

    def groupby_except(
        self, non_groupers: str | list[str], observed: bool = True
    ) -> pd.core.groupby.generic.SeriesGroupBy[Any, Any]:
        """
        Group by all index levels except specified levels

        This is the inverse of [pd.Series.groupby][pandas.Series.groupby].

        Parameters
        ----------
        non_groupers
            Columns to exclude from the grouping

        observed
            Whether to only return observed combinations or not

        Returns
        -------
        :
            The [pd.Series][pandas.Series],
            grouped by all columns except `non_groupers`.
        """
        return groupby_except(
            self._series, non_groupers=non_groupers, observed=observed
        )

    def mi_loc(
        self,
        locator: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector,
    ) -> S:
        """
        Select data, being slightly smarter than the default [pandas.Series.loc][].

        Parameters
        ----------
        locator
            Locator to apply

            If this is a multi-index, we use
            [multi_index_lookup][pandas_openscm.indexing.]
            to ensure correct alignment.

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
        return mi_loc(self._series, locator)

    def set_index_levels(
        self,
        levels_to_set: dict[str, Any | Collection[Any]],
        copy: bool = True,
    ) -> S:
        """
        Set the index levels

        Parameters
        ----------
        levels_to_set
            Mapping of level names to values to set

        copy
            Should the [pd.Series][pandas.Series] be copied before returning?

        Returns
        -------
        :
            [pd.Series][pandas.Series] with updates applied to its index
        """
        res = set_index_levels_func(
            self._series,
            levels_to_set=levels_to_set,
            copy=copy,
        )

        # Ignore return type
        # because I've done something wrong with how I've set this up.
        # Figuring this out is a job for another day
        return res  # type: ignore

    # def to_category_index(self) -> pd.DataFrame:
    #     """
    #     Convert the index's values to categories
    #
    #     This can save a lot of memory and improve the speed of processing.
    #     However, it comes with some pitfalls.
    #     For a nice discussion of some of them,
    #     see [this article](https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a/).
    #
    #     Returns
    #     -------
    #     :
    #         [pd.DataFrame][pandas.DataFrame] with all index columns
    #         converted to category type.
    #     """
    #     return convert_index_to_category_index(self._df)

    # def update_index_levels(
    #     self,
    #     updates: dict[Any, Callable[[Any], Any]],
    #     copy: bool = True,
    #     remove_unused_levels: bool = True,
    # ) -> pd.DataFrame:
    #     """
    #     Update the index levels
    #
    #     Parameters
    #     ----------
    #     updates
    #         Updates to apply to the index levels
    #
    #         Each key is the index level to which the updates will be applied.
    #         Each value is a function which updates the levels to their new values.
    #
    #     copy
    #         Should the [pd.DataFrame][pandas.DataFrame] be copied before returning?
    #
    #     remove_unused_levels
    #         Remove unused levels before applying the update
    #
    #         Specifically, call
    #         [pd.MultiIndex.remove_unused_levels][pandas.MultiIndex.remove_unused_levels].  # noqa: E501
    #
    #         This avoids trying to update levels that aren't being used.
    #
    #     Returns
    #     -------
    #     :
    #         [pd.DataFrame][pandas.DataFrame] with updates applied to its index
    #     """
    #     return update_index_levels_func(
    #         self._df,
    #         updates=updates,
    #         copy=copy,
    #         remove_unused_levels=remove_unused_levels,
    #     )
    #
    # def update_index_levels_from_other(
    #     self,
    #     update_sources: dict[
    #         Any, tuple[Any, Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any]]
    #     ],
    #     copy: bool = True,
    #     remove_unused_levels: bool = True,
    # ) -> pd.DataFrame:
    #     """
    #     Update the index levels based on other index levels
    #
    #     Parameters
    #     ----------
    #     update_sources
    #         Updates to apply to `df`'s index
    #
    #         Each key is the level to which the updates will be applied
    #         (or the level that will be created if it doesn't already exist).
    #
    #         Each value is a tuple of which the first element
    #         is the level to use to generate the values (the 'source level')
    #         and the second is mapper of the form used by
    #         [pd.Index.map][pandas.Index.map]
    #         which will be applied to the source level
    #         to update/create the level of interest.
    #
    #     copy
    #         Should the [pd.DataFrame][pandas.DataFrame] be copied before returning?
    #
    #     remove_unused_levels
    #         Remove unused levels before applying the update
    #
    #         Specifically, call
    #         [pd.MultiIndex.remove_unused_levels][pandas.MultiIndex.remove_unused_levels].  # noqa: E501
    #
    #         This avoids trying to update levels that aren't being used.
    #
    #     Returns
    #     -------
    #     :
    #         [pd.DataFrame][pandas.DataFrame] with updates applied to its index
    #     """
    #     return update_index_levels_from_other_func(
    #         self._df,
    #         update_sources=update_sources,
    #         copy=copy,
    #         remove_unused_levels=remove_unused_levels,
    #     )
