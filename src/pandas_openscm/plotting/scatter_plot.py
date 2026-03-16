"""
Generation of scatter plots

This reimplements seaborn in many ways,
but has been adjusted to suit the style of data we have.
If you're looking at this, you may also want to consider
raw seaborn (or even matplotlib), because those libraries
provide much more flexibility than this API.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    overload,
)

import pandas as pd
from attrs import define, field

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.indexing import mi_loc
from pandas_openscm.plotting.axis_labels import (
    cast_label_false_to_none,
    infer_label,
    try_to_get_unit_label,
)
from pandas_openscm.plotting.data_validation import is_same_shape_as_x_vals
from pandas_openscm.plotting.from_pandas_helpers import (
    fill_out_markers,
    fill_out_palette,
    get_default_color_var_label,
    get_default_marker_var_label,
)
from pandas_openscm.plotting.legend import create_legend_default

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.markers
    import pint

    from pandas_openscm.plotting.typing import (
        COLOUR_VALUE_LIKE,
        MARKER_VALUE_LIKE,
        PALETTE_LIKE,
    )
    from pandas_openscm.typing import NP_ARRAY_OF_FLOAT_OR_INT, PINT_NUMPY_ARRAY


@overload
def get_values_scatter(
    pseries: pd.Series,
    *,
    unit_aware: Literal[False],
    unit_var: str | None,
    stack_index_level: Any,
    x_stacked_column: Any,
    y_stacked_column: Any,
) -> tuple[NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT]: ...


@overload
def get_values_scatter(
    pseries: pd.Series,
    *,
    unit_aware: Literal[True] | pint.facets.PlainRegistry,
    unit_var: str | None,
    stack_index_level: Any,
    x_stacked_column: Any,
    y_stacked_column: Any,
) -> tuple[PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY]: ...


def get_values_scatter(  # noqa: PLR0913
    pseries: pd.Series,
    *,
    unit_aware: bool | pint.facets.PlainRegistry,
    unit_var: str | None,
    stack_index_level: Any,
    x_stacked_column: Any,
    y_stacked_column: Any,
) -> (
    tuple[NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT]
    | tuple[PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY]
):
    """
    Get values for plotting a scatter

    Parameters
    ----------
    pseries
        [pd.Series][pandas.Series] from which to get the values

    unit_aware
        Should the values be unit-aware?

        If `True`, we use the default application registry
        (retrieved with [pint.get_application_registry][]).
        Otherwise, a [pint.facets.PlainRegistry][] can be supplied and will be used.

    unit_var
        Variable/column in the multi-index which stores information
        about the unit of the data.

    stack_index_level
        Index level to stack

    x_stacked_column
        Column to use for x-values after stacking `stack_index_level`.

    y_stacked_column
        Column to use for y-values after stacking `stack_index_level`.

    Returns
    -------
    x_values :
        x-values (for a plot)

    y_values :
        y-values (for a plot)

    Raises
    ------
    TypeError
        `unit_aware` is not `False` and `unit_var`.

    MissingOptionalDependencyError
        `unit_aware` is `True`
        and [pint](https://pint.readthedocs.io/) is not installed.
    """
    pseries_stacked = pseries.reset_index(unit_var, drop=True).unstack(
        stack_index_level
    )
    res_no_units = (
        pseries_stacked[x_stacked_column].values.squeeze(),
        pseries_stacked[y_stacked_column].values.squeeze(),
    )
    if not unit_aware:
        return res_no_units

    if unit_var is None:
        msg = "If `unit_aware` != False, then `unit_var` must not be `None`"
        raise TypeError(msg)

    if isinstance(unit_aware, bool):
        try:
            import pint
        except ImportError as exc:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "get_values_scatter(..., unit_aware=True, ...)", requirement="pint"
            ) from exc

        ur = pint.get_application_registry()  # type: ignore

    else:
        ur = unit_aware

    x_units = (
        mi_loc(pseries, pd.Index([x_stacked_column], name=stack_index_level))
        .index.get_level_values(unit_var)
        .unique()
    )
    if len(x_units) == 1:
        x_unit = x_units[0]
    else:
        msg = "more than one unit for the x-values"
        raise NotImplementedError(msg)

    y_units = (
        mi_loc(pseries, pd.Index([y_stacked_column], name=stack_index_level))
        .index.get_level_values(unit_var)
        .unique()
    )
    if len(y_units) == 1:
        y_unit = y_units[0]
    else:
        msg = "more than one unit for the y-values"
        raise NotImplementedError(msg)

    res = (
        res_no_units[0] * ur(x_unit),
        res_no_units[1] * ur(y_unit),
    )

    return res


def get_axis_label(
    stacked_column: Any,
    label_in: str | bool | None,
    pseries: pd.Series[Any],
    unit_index_level: str,
    warn_infer_label_with_multi_unit: bool,
) -> str | None:
    """
    Get axis label

    Parameters
    ----------
    stacked_column
        Stacked column being plotted

    label_in
        Input value of the label.

        If a `str`, `label_in` is simply returned.

        If `True`, we will try and infer the label based on the data's units.
        If we can get the units, the label will combine the column being plotted
        and the units.
        If we can't get the units, the label will simply be the column being plotted.

        If `None` or `False`, `None` is returned.

    pseries
        [pd.Series][pandas.Series] being plotted

    unit_index_level
        Level in `pseries.index` which contains unit information

    warn_infer_label_with_multi_unit
        Should a warning be raised if we try to infer the unit
        but the data has more than one unit?

    Returns
    -------
    :
        Derived label
    """
    if infer_label(label_in):
        label_units = try_to_get_unit_label(
            # No unit-aware plotting for scatter plots,
            # always want x_stacked_column and y_stacked_column in axis labels
            # by default.
            unit_aware=False,
            pandas_obj=pseries,
            unit_index_level=unit_index_level,
            # TODO: think about how to handle this warning better
            warn_infer_label_with_multi_unit=warn_infer_label_with_multi_unit,
        )
        if label_units is not None:
            res = f"{stacked_column} [{label_units}]"
        else:
            res = stacked_column

    else:
        res = cast_label_false_to_none(label_in)

    return res


@define
class SingleScatterPlotter:
    """Object which is able to plot single scatters"""

    x_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY
    """x-values to plot"""

    y_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY = field(
        validator=[is_same_shape_as_x_vals]
    )
    """y-values to plot"""

    marker: MARKER_VALUE_LIKE
    """Marker to use when plotting the scatter"""

    size: float
    """Size of marker to use when plotting"""

    color: COLOUR_VALUE_LIKE
    """Colour to use when plotting"""

    alpha: float
    """Alpha to use when plotting"""

    pkwargs: dict[str, Any] | None = None
    """Other arguments to pass to [matplotlib.axes.Axes.scatter][] when plotting"""

    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """
        Plot

        Parameters
        ----------
        ax
            Axes on which to plot
        """
        pkwargs = self.pkwargs if self.pkwargs is not None else {}

        ax.scatter(
            self.x_vals,
            self.y_vals,
            marker=self.marker,
            s=self.size,
            c=self.color,
            alpha=self.alpha,
            **pkwargs,
        )

        return ax


@define
class SeabornLikeScatterPlotter:
    """
    Seaborn-like plotter for scatter plots

    This is really just a data holder,
    which allows us to split the logic for preparing data
    from the logic of actually making plots.
    This is useful because we don't want all the individual scatters
    to appear in the legend, rather only summaries of the hue and marker
    used for each scatter
    (achieving such behaviour with the available matplotlib API is difficult).

    If you use this class directly, be careful.
    It is easy to create inconsistencies between the scatters to be plotted
    and the other information (which is used to create the legend entries).
    For example, if you alter `self.palette`
    without altering `self.scatters` accordingly,
    you will get legend entries that don't correspond to any lines.

    It's 'seaborn-like' because it is based on similar ideas to
    [seaborn](https://seaborn.pydata.org/),
    but has been adjusted to suit the style of data we have.
    If you're looking at this, you may also want to consider
    raw seaborn (or even matplotlib), because those libraries
    provide much more flexibility than this API.
    """

    scatters: Iterable[SingleScatterPlotter]
    """Scatters plotters"""

    color_var_label: str
    """Label for the variable by which scatters are coloured in the legend"""

    marker_var_label: str | None
    """
    Label for the variable by which scatters are styled in the legend (if not `None`)
    """

    palette: PALETTE_LIKE[Any]
    """
    Palette used for different values of the variable by which scatters are coloured
    """

    markers: dict[Any, MARKER_VALUE_LIKE] | None
    """
    Markers used for different values of the variable by which scatters are styled
    """

    x_label: str | None
    """Label to apply to the x-axis (if `None`, no label is applied)"""

    y_label: str | None
    """Label to apply to the y-axis (if `None`, no label is applied)"""

    @classmethod
    def from_series(  # noqa: PLR0913 # object creation code is the worst
        cls,
        series: pd.Series,
        *,
        stack_index_level: Any,
        x_stacked_column: Any,
        y_stacked_column: Any,
        color_var: str = "scenario",
        color_var_label: str | None = None,
        palette: PALETTE_LIKE[Any] | None = None,
        warn_on_palette_value_missing: bool = True,
        marker_var: str | None = None,
        marker_var_label: str | None = None,
        markers: dict[Any, MARKER_VALUE_LIKE] | None = None,
        warn_on_marker_value_missing: bool = True,
        size: float = 30.0,
        alpha: float = 0.8,
        unit_var: str | None = "unit",
        unit_aware: bool | pint.facets.PlainRegistry = False,
        x_label: str | bool | None = True,
        warn_infer_x_label_with_multi_unit: bool = True,
        y_label: str | bool | None = True,
        warn_infer_y_label_with_multi_unit: bool = True,
        observed: bool = True,
    ) -> SeabornLikeScatterPlotter:
        """
        Initialise from a [pd.DataFrame][pandas.DataFrame]

        Parameters
        ----------
        series
            [pd.Series][pandas.Series] from which to initialise

        stack_index_level
            Index level to stack

        x_stacked_column
            Column to use as the x-axis

            This value should be a value from `stack_index_level`
            (so will be a column name once `series` is unstacked).

        y_stacked_column
            Column to use as the y-axis

            This value should be a value from `stack_index_level`
            (so will be a column name once `series` is unstacked).

        color_var
            Variable to use for grouping data into different colour groups

        color_var_label
            Label to use as the header for the colour section in the legend

        palette
            Colour to use for the different groups in the data.

            If any groups are not included in `palette`,
            they are auto-filled.

        warn_on_palette_value_missing
            Should a warning be emitted if there are values missing from `palette`?

        marker_var
            Variable to use for grouping data into different marker groups

        marker_var_label
            Label to use as the header for the marker section in the legend

        markers
            Markers to use for the different groups in the data.

            If any groups are not included in `markers`,
            they are auto-filled.

        warn_on_marker_value_missing
            Should a warning be emitted if there are values missing from `markers`?

        size
            Size of marker to use for plotting scatters.

        alpha
            Alpha to use when plotting the scatters

        unit_var
            Variable/column in the multi-index which stores information
            about the unit of each timeseries.

        unit_aware
            Should the values be extracted in a unit-aware way?

            If `True`, we use the default application registry
            (retrieved with [pint.get_application_registry][]).
            Otherwise, a [pint.facets.PlainRegistry][] can be supplied and will be used.

        x_label
            Label to apply to the x-axis.

            If `True`, we will try and infer the x-label based on the data's units.

            If `None` or `False`, no label will be applied.

        warn_infer_x_label_with_multi_unit
            Should a warning be raised if we try to infer the x-unit
            but the data has more than one unit?

        y_label
            Label to apply to the y-axis.

            If `True`, we will try and infer the y-label based on the data's units.

            If `None` or `False`, no label will be applied.

        warn_infer_y_label_with_multi_unit
            Should a warning be raised if we try to infer the y-unit
            but the data has more than one unit?

        observed
            Passed to [pd.DataFrame.groupby][pandas.DataFrame.groupby].

        Returns
        -------
        :
             Initialised instance
        """
        if color_var_label is None:
            color_var_label = get_default_color_var_label(color_var)

        if marker_var is not None and marker_var_label is None:
            marker_var_label = get_default_marker_var_label(marker_var)

        palette_complete = fill_out_palette(
            series,
            color_index_level=color_var,
            palette_user_supplied=palette,
            warn_on_value_missing=warn_on_palette_value_missing,
        )

        if marker_var is not None:
            group_cols = [color_var, marker_var]
            markers_complete = fill_out_markers(
                series,
                marker_index_level=marker_var,
                markers_user_supplied=markers,
                warn_on_value_missing=warn_on_marker_value_missing,
            )

        else:
            group_cols = [color_var]
            markers_complete = None

        scatters: list[SingleScatterPlotter] = []
        for info, gseries in series.groupby(group_cols, observed=observed):
            info_d = {k: v for k, v in zip(group_cols, info)}

            colour = palette_complete[info_d[color_var]]

            if marker_var is not None:
                if markers_complete is None:  # pragma: no cover
                    # should be impossible to hit this
                    raise AssertionError
                marker = markers_complete[info_d[marker_var]]

            else:
                marker = "x"

            scatter_plotter = SingleScatterPlotter(
                *get_values_scatter(
                    gseries,
                    unit_aware=unit_aware,  # type: ignore # not sure why mypy is complaining
                    unit_var=unit_var,
                    stack_index_level=stack_index_level,
                    x_stacked_column=x_stacked_column,
                    y_stacked_column=y_stacked_column,
                ),
                marker=marker,
                size=size,
                color=colour,
                alpha=alpha,
            )
            scatters.append(scatter_plotter)

        series_x_relevant = mi_loc(
            series, pd.Index([x_stacked_column], name=stack_index_level)
        )
        series_y_relevant = mi_loc(
            series, pd.Index([y_stacked_column], name=stack_index_level)
        )

        x_label = get_axis_label(
            stacked_column=x_stacked_column,
            label_in=x_label,
            pseries=series_x_relevant,
            unit_index_level=unit_var,
            warn_infer_label_with_multi_unit=warn_infer_x_label_with_multi_unit,
        )

        y_label = get_axis_label(
            stacked_column=y_stacked_column,
            label_in=y_label,
            pseries=series_y_relevant,
            unit_index_level=unit_var,
            warn_infer_label_with_multi_unit=warn_infer_y_label_with_multi_unit,
        )

        res = SeabornLikeScatterPlotter(
            scatters=scatters,
            color_var_label=color_var_label,
            marker_var_label=marker_var_label,
            palette=palette_complete,
            markers=markers_complete,
            x_label=x_label,
            y_label=y_label,
        )

        return res

    def generate_legend_handles(self) -> list[matplotlib.artist.Artist]:
        """
        Generate handles for the legend

        Returns
        -------
        :
            Generated handles for the legend
        """
        try:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "generate_legend_handles", requirement="matplotlib"
            ) from exc

        color_items = [
            mlines.Line2D([], [], color=color, label=color_value)
            for color_value, color in self.palette.items()
        ]
        legend_items = [
            mpatches.Patch(alpha=0, label=self.color_var_label),
            *color_items,
        ]
        if self.markers is not None:
            marker_items = [
                mlines.Line2D(
                    [],
                    [],
                    color="w",
                    marker=marker,
                    label=marker_value,
                    markerfacecolor="gray",
                    markeredgecolor="gray",
                    markeredgewidth=1.0,
                    # markersize=30,
                )
                for marker_value, marker in self.markers.items()
            ]
            legend_items.append(mpatches.Patch(alpha=0, label=self.marker_var_label))
            legend_items.extend(marker_items)

        return legend_items

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        *,
        create_legend: Callable[
            [matplotlib.axes.Axes, list[matplotlib.artist.Artist]], None
        ] = create_legend_default,
    ) -> matplotlib.axes.Axes:
        """
        Plot

        Parameters
        ----------
        ax
            Axes onto which to plot

        create_legend
            Function to use to create the legend.

            This allows the user to have full control over the creation of the legend.

        Returns
        -------
        :
            Axes on which the data was plotted
        """
        if ax is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise MissingOptionalDependencyError(  # noqa: TRY003
                    "plot(ax=None, ...)", requirement="matplotlib"
                ) from exc

            _, ax = plt.subplots()

        for scatter in self.scatters:
            scatter.plot(ax=ax)

        create_legend(ax, self.generate_legend_handles())

        if self.x_label is not None:
            ax.set_xlabel(self.x_label)

        if self.y_label is not None:
            ax.set_ylabel(self.y_label)

        return ax
