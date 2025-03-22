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
from pandas_openscm.index_manipulation import convert_index_to_category_index
from pandas_openscm.indexing import mi_loc
from pandas_openscm.plotting import (
    create_legend_default,
    plot_plume,
    plot_plume_after_calculating_quantiles,
)

if TYPE_CHECKING:
    import matplotlib
    import pandas_indexing as pix

    from pandas_openscm.plotting import COLOUR_VALUE_LIKE, QUANTILES_PLUMES_LIKE


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
    ) -> pd.core.groupby.generic.DataFrameGroupBy:
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

    def plot_plume(
        self,
        # TODO: match plot_plume API
        ax: matplotlib.axes.Axes | None = None,
        *,
        quantiles_plumes: QUANTILES_PLUMES_LIKE = (
            (0.5, 0.7),
            ((0.05, 0.95), 0.2),
        ),
        quantile_var: str = "quantile",
        quantile_var_label: str | None = None,
        quantile_legend_round: int = 2,
        hue_var: str = "scenario",
        hue_var_label: str | None = None,
        palette: dict[Any, COLOUR_VALUE_LIKE | tuple[COLOUR_VALUE_LIKE, float]]
        | None = None,
        warn_on_palette_value_missing: bool = True,
        style_var: str = "variable",
        style_var_label: str | None = None,
        dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None = None,
        warn_on_dashes_value_missing: bool = True,
        linewidth: float = 3.0,
        unit_col: str = "unit",
        x_label: str | None = "time",
        y_label: str | bool | None = True,
        warn_infer_y_label_with_multi_unit: bool = True,
        create_legend: Callable[
            [matplotlib.axes.Axes, list[matplotlib.artist.Artist]], None
        ] = create_legend_default,
        observed: bool = True,
    ) -> matplotlib.axes.Axes:
        # TODO: docstring
        return plot_plume(
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
            unit_col=unit_col,
            x_label=x_label,
            y_label=y_label,
            warn_infer_y_label_with_multi_unit=warn_infer_y_label_with_multi_unit,
            create_legend=create_legend,
            observed=observed,
        )

    def plot_plume_after_calculating_quantiles(
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
        palette: dict[Any, COLOUR_VALUE_LIKE | tuple[COLOUR_VALUE_LIKE, float]]
        | None = None,
        warn_on_palette_value_missing: bool = True,
        style_var: str = "variable",
        style_var_label: str | None = None,
        dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None = None,
        warn_on_dashes_value_missing: bool = True,
        linewidth: float = 3.0,
        unit_col: str = "unit",
        x_label: str | None = "time",
        y_label: str | bool | None = True,
        warn_infer_y_label_with_multi_unit: bool = True,
        create_legend: Callable[
            [matplotlib.axes.Axes, list[matplotlib.artist.Artist]], None
        ] = create_legend_default,
        observed: bool = True,
    ) -> matplotlib.axes.Axes:
        # TODO: docstring
        return plot_plume_after_calculating_quantiles(
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
            unit_col=unit_col,
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
