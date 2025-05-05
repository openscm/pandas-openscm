"""
API for [pandas][] accessors.

As a general note to developers,
we try and keep the accessors as a super-thin layer.
This makes it easier to re-use functionality in a more functional way,
which is beneficial
(particularly if we one day need to switch to
a different kind of dataframe e.g. dask).

As a result, we effectively duplicate our API.
This is fine, because this repo is not so big.
Pandas and pandas-indexing use pandas' `pandas.util._decorators.docs` decorator
(see https://github.com/pandas-dev/pandas/blob/05de25381f71657bd425d2c4045d81a46b2d3740/pandas/util/_decorators.py#L342)
to avoid duplicating the docs.
We could use the same pattern, but I have found that this magic
almost always goes wrong so I would stay away from this as long as we can.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import (
    convert_index_to_category_index,
    update_index_levels_from_other_func,
    update_index_levels_func,
)
from pandas_openscm.indexing import mi_loc
from pandas_openscm.plotting import (
    create_legend_default,
    plot_plume_after_calculating_quantiles_func,
    plot_plume_func,
)
from pandas_openscm.reshaping import ts_to_long_data

if TYPE_CHECKING:
    import matplotlib
    import pandas_indexing as pix
    import pint

    from pandas_openscm.plotting import (
        PALETTE_LIKE,
        QUANTILES_PLUMES_LIKE,
    )


class DataFramePandasOpenSCMAccessor:
    """
    [pd.DataFrame][pandas.DataFrame] accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialise

        Parameters
        ----------
        pandas_obj
            Pandas object to use via the accessor
        """
        # It is possible to validate here.
        # However, it's probably better to do validation closer to the data use.
        self._df = pandas_obj

    def fix_index_name_after_groupby_quantile(
        self, new_name: str = "quantile", copy: bool = False
    ) -> pd.DataFrame:
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
            Whether to copy `df` before manipulating the index name

        Returns
        -------
        :
            `df`, with the last level in its index renamed to `new_name`.
        """
        return fix_index_name_after_groupby_quantile(
            self._df, new_name=new_name, copy=copy
        )

    def groupby_except(
        self, non_groupers: str | list[str], observed: bool = True
    ) -> pd.core.groupby.generic.DataFrameGroupBy[Any]:
        """
        Group by all index levels except specified levels

        This is the inverse of [pd.DataFrame.groupby][pandas.DataFrame.groupby].

        Parameters
        ----------
        non_groupers
            Columns to exclude from the grouping

        observed
            Whether to only return observed combinations or not

        Returns
        -------
        :
            The [pd.DataFrame][pandas.DataFrame],
            grouped by all columns except `non_groupers`.
        """
        return groupby_except(df=self._df, non_groupers=non_groupers, observed=observed)

    def mi_loc(
        self,
        locator: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector,
    ) -> pd.DataFrame:
        """
        Select data, being slightly smarter than the default [pandas.DataFrame.loc][].

        Parameters
        ----------
        locator
            Locator to apply

            If this is a multi-index, we use
            [multi_index_lookup][(p).indexing.] to ensure correct alignment.

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
        return mi_loc(self._df, locator)

    def plot_plume(  # noqa: PLR0913
        self,
        quantiles_plumes: QUANTILES_PLUMES_LIKE,
        ax: matplotlib.axes.Axes | None = None,
        *,
        quantile_var: str = "quantile",
        quantile_var_label: str | None = None,
        quantile_legend_round: int = 3,
        hue_var: str = "scenario",
        hue_var_label: str | None = None,
        palette: PALETTE_LIKE[Any] | None = None,
        warn_on_palette_value_missing: bool = True,
        style_var: str = "variable",
        style_var_label: str | None = None,
        dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None = None,
        warn_on_dashes_value_missing: bool = True,
        linewidth: float = 2.0,
        unit_var: str = "unit",
        unit_aware: bool | pint.facets.PlainRegistry = False,
        time_units: str | None = None,
        x_label: str | None = "time",
        y_label: str | bool | None = True,
        warn_infer_y_label_with_multi_unit: bool = True,
        create_legend: Callable[
            [matplotlib.axes.Axes, list[matplotlib.artist.Artist]], None
        ] = create_legend_default,
        observed: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot a plume plot

        Parameters
        ----------
        quantiles_plumes
            Quantiles to plot in each plume.

            If the first element of each tuple is a tuple,
            a plume is plotted between the given quantiles.
            Otherwise, if the first element is a plain float,
            a line is plotted for the given quantile.

        ax
            Axes on which to plot.

            If not supplied, a new axes is created.

        quantile_var
            Variable/column in the multi-index which stores information
            about the quantile that each timeseries represents.

        quantile_var_label
            Label to use as the header for the quantile section in the legend

        quantile_legend_round
            Rounding to apply to quantile values when creating the legend

        hue_var
            Variable to use for grouping data into different colour groups

        hue_var_label
            Label to use as the header for the hue/colour section in the legend

        palette
            Colour to use for the different groups in the data.

            If any groups are not included in `palette`,
            they are auto-filled.

        warn_on_palette_value_missing
            Should a warning be emitted if there are values missing from `palette`?

        style_var
            Variable to use for grouping data into different (line)style groups

        style_var_label
            Label to use as the header for the style section in the legend

        dashes
            Dash/linestyle to use for the different groups in the data.

            If any groups are not included in `dashes`,
            they are auto-filled.

        warn_on_dashes_value_missing
            Should a warning be emitted if there are values missing from `dashes`?

        linewidth
            Width to use for plotting lines.

        unit_var
            Variable/column in the multi-index which stores information
            about the unit of each timeseries.

        unit_aware
            Should the plot be done in a unit-aware way?

            If `True`, we use the default application registry
            (retrieved with [pint.get_application_registry][]).
            Otherwise, a [pint.facets.PlainRegistry][] can be supplied and will be used.

            For details, see matplotlib and pint support plotting with units
            ([stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html),
            [last version that we checked at the time of writing](https://pint.readthedocs.io/en/0.24.4/user/plotting.html)).

        time_units
            Units of the time axis of the data.

            These are required if `unit_aware` is not `False`.

        x_label
            Label to apply to the x-axis.

            If `None`, no label will be applied.

        y_label
            Label to apply to the y-axis.

            If `True`, we will try and infer the y-label based on the data's units.

            If `None`, no label will be applied.

        warn_infer_y_label_with_multi_unit
            Should a warning be raised if we try to infer the y-unit
            but the data has more than one unit?

        create_legend
            Function to use to create the legend.

            This allows the user to have full control over the creation of the legend.

        observed
            Passed to [pd.DataFrame.groupby][pandas.DataFrame.groupby].

        Returns
        -------
        :
            Axes on which the data was plotted
        """
        return plot_plume_func(
            self._df,
            ax=ax,
            quantiles_plumes=quantiles_plumes,
            quantile_var=quantile_var,
            quantile_var_label=quantile_var_label,
            quantile_legend_round=quantile_legend_round,
            hue_var=hue_var,
            hue_var_label=hue_var_label,
            palette=palette,
            warn_on_palette_value_missing=warn_on_palette_value_missing,
            style_var=style_var,
            style_var_label=style_var_label,
            dashes=dashes,
            warn_on_dashes_value_missing=warn_on_dashes_value_missing,
            linewidth=linewidth,
            unit_var=unit_var,
            unit_aware=unit_aware,
            time_units=time_units,
            x_label=x_label,
            y_label=y_label,
            warn_infer_y_label_with_multi_unit=warn_infer_y_label_with_multi_unit,
            create_legend=create_legend,
            observed=observed,
        )

    def plot_plume_after_calculating_quantiles(  # noqa: PLR0913
        self,
        ax: matplotlib.axes.Axes | None = None,
        *,
        quantile_over: str | list[str],
        quantiles_plumes: QUANTILES_PLUMES_LIKE = (
            (0.5, 0.7),
            ((0.05, 0.95), 0.2),
        ),
        quantile_var_label: str | None = None,
        quantile_legend_round: int = 2,
        hue_var: str = "scenario",
        hue_var_label: str | None = None,
        palette: PALETTE_LIKE[Any] | None = None,
        warn_on_palette_value_missing: bool = True,
        style_var: str = "variable",
        style_var_label: str | None = None,
        dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None = None,
        warn_on_dashes_value_missing: bool = True,
        linewidth: float = 3.0,
        unit_var: str = "unit",
        unit_aware: bool | pint.facets.PlainRegistry = False,
        time_units: str | None = None,
        x_label: str | None = "time",
        y_label: str | bool | None = True,
        warn_infer_y_label_with_multi_unit: bool = True,
        create_legend: Callable[
            [matplotlib.axes.Axes, list[matplotlib.artist.Artist]], None
        ] = create_legend_default,
        observed: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot a plume plot, calculating the required quantiles first

        Parameters
        ----------
        ax
            Axes on which to plot.

            If not supplied, a new axes is created.

        quantile_over
            Variable(s)/column(s) over which to calculate the quantiles.

            The data is grouped by all columns except `quantile_over`
            when calculating the quantiles.

        quantiles_plumes
            Quantiles to plot in each plume.

            If the first element of each tuple is a tuple,
            a plume is plotted between the given quantiles.
            Otherwise, if the first element is a plain float,
            a line is plotted for the given quantile.

        quantile_var_label
            Label to use as the header for the quantile section in the legend

        quantile_legend_round
            Rounding to apply to quantile values when creating the legend

        hue_var
            Variable to use for grouping data into different colour groups

        hue_var_label
            Label to use as the header for the hue/colour section in the legend

        palette
            Colour to use for the different groups in the data.

            If any groups are not included in `palette`,
            they are auto-filled.

        warn_on_palette_value_missing
            Should a warning be emitted if there are values missing from `palette`?

        style_var
            Variable to use for grouping data into different (line)style groups

        style_var_label
            Label to use as the header for the style section in the legend

        dashes
            Dash/linestyle to use for the different groups in the data.

            If any groups are not included in `dashes`,
            they are auto-filled.

        warn_on_dashes_value_missing
            Should a warning be emitted if there are values missing from `dashes`?

        linewidth
            Width to use for plotting lines.

        unit_var
            Variable/column in the multi-index which stores information
            about the unit of each timeseries.

        unit_aware
            Should the plot be done in a unit-aware way?

            If `True`, we use the default application registry
            (retrieved with [pint.get_application_registry][]).
            Otherwise, a [pint.facets.PlainRegistry][] can be supplied and will be used.

            For details, see matplotlib and pint support plotting with units
            ([stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html),
            [last version that we checked at the time of writing](https://pint.readthedocs.io/en/0.24.4/user/plotting.html)).

        time_units
            Units of the time axis.

            These are required if `unit_aware` is not `False`.

        x_label
            Label to apply to the x-axis.

            If `None`, no label will be applied.

        y_label
            Label to apply to the y-axis.

            If `True`, we will try and infer the y-label based on the data's units.

            If `None`, no label will be applied.

        warn_infer_y_label_with_multi_unit
            Should a warning be raised if we try to infer the y-unit
            but the data has more than one unit?

        create_legend
            Function to use to create the legend.

            This allows the user to have full control over the creation of the legend.

        observed
            Passed to [pd.DataFrame.groupby][pandas.DataFrame.groupby].

        Returns
        -------
        :
            Axes on which the data was plotted
        """
        return plot_plume_after_calculating_quantiles_func(
            self._df,
            ax=ax,
            quantile_over=quantile_over,
            quantiles_plumes=quantiles_plumes,
            quantile_var_label=quantile_var_label,
            quantile_legend_round=quantile_legend_round,
            hue_var=hue_var,
            hue_var_label=hue_var_label,
            palette=palette,
            warn_on_palette_value_missing=warn_on_palette_value_missing,
            style_var=style_var,
            style_var_label=style_var_label,
            dashes=dashes,
            warn_on_dashes_value_missing=warn_on_dashes_value_missing,
            linewidth=linewidth,
            unit_var=unit_var,
            unit_aware=unit_aware,
            time_units=time_units,
            x_label=x_label,
            y_label=y_label,
            warn_infer_y_label_with_multi_unit=warn_infer_y_label_with_multi_unit,
            create_legend=create_legend,
            observed=observed,
        )

    def to_category_index(self) -> pd.DataFrame:
        """
        Convert the index's values to categories

        This can save a lot of memory and improve the speed of processing.
        However, it comes with some pitfalls.
        For a nice discussion of some of them,
        see [this article](https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a/).

        Returns
        -------
        :
            [pd.DataFrame][pandas.DataFrame] with all index columns
            converted to category type.
        """
        return convert_index_to_category_index(self._df)

    def to_long_data(self, time_col_name: str = "time") -> pd.DataFrame:
        """
        Convert to long data

        Here, long data means that each row contains a single value,
        alongside metadata associated with that value
        (for more details, see e.g.
        https://data.europa.eu/apps/data-visualisation-guide/wide-versus-long-data).

        Parameters
        ----------
        time_col_name
            Name of the time column in the output

        Returns
        -------
        :
            DataFrame in long-form

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>>
        >>> from pandas_openscm.accessors import register_pandas_accessor
        >>>
        >>> register_pandas_accessor()
        >>>
        >>> df = pd.DataFrame(
        ...     [
        ...         [1.1, 0.8, 1.2],
        ...         [2.1, np.nan, 8.4],
        ...         [2.3, 3.2, 3.0],
        ...         [1.2, 2.8, np.nan],
        ...     ],
        ...     columns=[2010.0, 2015.0, 2025.0],
        ...     index=pd.MultiIndex.from_tuples(
        ...         [
        ...             ("sa", np.nan, "K"),
        ...             ("sb", "v1", None),
        ...             ("sa", "v2", "W"),
        ...             ("sb", "v2", "W"),
        ...         ],
        ...         names=["scenario", "variable", "unit"],
        ...     ),
        ... )
        >>>
        >>> # Start with wide data
        >>> df
                                2010.0  2015.0  2025.0
        scenario variable unit
        sa       NaN      K        1.1     0.8     1.2
        sb       v1       NaN      2.1     NaN     8.4
        sa       v2       W        2.3     3.2     3.0
        sb       v2       W        1.2     2.8     NaN
        >>>
        >>> # Convert to long data
        >>> df.openscm.to_long_data()
           scenario variable unit    time  value
        0        sa      NaN    K  2010.0    1.1
        1        sb       v1  NaN  2010.0    2.1
        2        sa       v2    W  2010.0    2.3
        3        sb       v2    W  2010.0    1.2
        4        sa      NaN    K  2015.0    0.8
        5        sb       v1  NaN  2015.0    NaN
        6        sa       v2    W  2015.0    3.2
        7        sb       v2    W  2015.0    2.8
        8        sa      NaN    K  2025.0    1.2
        9        sb       v1  NaN  2025.0    8.4
        10       sa       v2    W  2025.0    3.0
        11       sb       v2    W  2025.0    NaN
        >>>
        >>> # Specify a different time column name
        >>> df.openscm.to_long_data(time_col_name="year")
           scenario variable unit    year  value
        0        sa      NaN    K  2010.0    1.1
        1        sb       v1  NaN  2010.0    2.1
        2        sa       v2    W  2010.0    2.3
        3        sb       v2    W  2010.0    1.2
        4        sa      NaN    K  2015.0    0.8
        5        sb       v1  NaN  2015.0    NaN
        6        sa       v2    W  2015.0    3.2
        7        sb       v2    W  2015.0    2.8
        8        sa      NaN    K  2025.0    1.2
        9        sb       v1  NaN  2025.0    8.4
        10       sa       v2    W  2025.0    3.0
        11       sb       v2    W  2025.0    NaN
        >>>
        >>> # The result is just a pandas DataFrame,
        >>> # so you can do whatever operations you want
        >>> # on the result.
        >>> # A common one is probably dropping all rows with NaN
        >>> df.openscm.to_long_data(time_col_name="year").dropna()
           scenario variable unit    year  value
        2        sa       v2    W  2010.0    2.3
        3        sb       v2    W  2010.0    1.2
        6        sa       v2    W  2015.0    3.2
        7        sb       v2    W  2015.0    2.8
        10       sa       v2    W  2025.0    3.0
        >>>
        >>> # or just rows with NaN in particular columns
        >>> df.openscm.to_long_data(time_col_name="year").dropna(subset=["variable"])
           scenario variable unit    year  value
        1        sb       v1  NaN  2010.0    2.1
        2        sa       v2    W  2010.0    2.3
        3        sb       v2    W  2010.0    1.2
        5        sb       v1  NaN  2015.0    NaN
        6        sa       v2    W  2015.0    3.2
        7        sb       v2    W  2015.0    2.8
        9        sb       v1  NaN  2025.0    8.4
        10       sa       v2    W  2025.0    3.0
        11       sb       v2    W  2025.0    NaN
        """
        return ts_to_long_data(self._df, time_col_name=time_col_name)

    def update_index_levels(
        self,
        updates: dict[Any, Callable[[Any], Any]],
        copy: bool = True,
        remove_unused_levels: bool = True,
    ) -> pd.DataFrame:
        """
        Update the index levels

        Parameters
        ----------
        updates
            Updates to apply to the index levels

            Each key is the index level to which the updates will be applied.
            Each value is a function which updates the levels to their new values.

        copy
            Should the [pd.DataFrame][pandas.DataFrame] be copied before returning?

        remove_unused_levels
            Remove unused levels before applying the update

            Specifically, call
            [pd.MultiIndex.remove_unused_levels][pandas.MultiIndex.remove_unused_levels].

            This avoids trying to update levels that aren't being used.

        Returns
        -------
        :
            [pd.DataFrame][pandas.DataFrame] with updates applied to its index
        """
        return update_index_levels_func(
            self._df,
            updates=updates,
            copy=copy,
            remove_unused_levels=remove_unused_levels,
        )

    def update_index_levels_from_other(
        self,
        update_sources: dict[
            Any, tuple[Any, Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any]]
        ],
        copy: bool = True,
        remove_unused_levels: bool = True,
    ) -> pd.DataFrame:
        """
        Update the index levels based on other index levels

        Parameters
        ----------
        update_sources
            Updates to apply to `df`'s index

            Each key is the level to which the updates will be applied
            (or the level that will be created if it doesn't already exist).

            Each value is a tuple of which the first element
            is the level to use to generate the values (the 'source level')
            and the second is mapper of the form used by
            [pd.Index.map][pandas.Index.map]
            which will be applied to the source level
            to update/create the level of interest.

        copy
            Should the [pd.DataFrame][pandas.DataFrame] be copied before returning?

        remove_unused_levels
            Remove unused levels before applying the update

            Specifically, call
            [pd.MultiIndex.remove_unused_levels][pandas.MultiIndex.remove_unused_levels].

            This avoids trying to update levels that aren't being used.

        Returns
        -------
        :
            [pd.DataFrame][pandas.DataFrame] with updates applied to its index
        """
        return update_index_levels_from_other_func(
            self._df,
            update_sources=update_sources,
            copy=copy,
            remove_unused_levels=remove_unused_levels,
        )


def register_pandas_accessor(namespace: str = "openscm") -> None:
    """
    Register the pandas accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

    We provide this as a separate function
    because we have had really bad experiences with imports having side effects
    and don't want to pass those on to our users.

    Parameters
    ----------
    namespace
        Namespace to use for the accessor
    """
    pd.api.extensions.register_dataframe_accessor(namespace)(
        DataFramePandasOpenSCMAccessor
    )
