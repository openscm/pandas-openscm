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
    handle_axis_label_inference_from_unit_information,
)
from pandas_openscm.plotting.data_validation import is_same_shape_as_x_vals
from pandas_openscm.plotting.from_pandas_helpers import (
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

    if time_units is None:
        msg = "If `unit_aware` != False, then `time_units` must not be `None`"
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

    res = (
        res_no_units[0] * ur(time_units),
        res_no_units[1] * ur(extract_single_unit(pdf, unit_var)),
    )

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

    marker: str | matplotlib.markers.MarkerStyle
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

    markers: dict[Any, str | matplotlib.markers.MarkerStyle] | None
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
        markers: dict[Any, str | matplotlib.markers.MarkerStyle] | None = None,
        warn_on_marker_value_missing: bool = True,
        size: float = 30.0,
        alpha: float = 0.8,
        unit_var: str | None = "unit",
        unit_aware: bool | pint.facets.PlainRegistry = False,
        time_units: str | None = None,
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

        time_units
            Units of the time axis of the data.

            These are required if `unit_aware` is not `False`.

        x_label
            Label to apply to the x-axis.

            If `True`, we will try and infer the x-label based on the data's units.

            If `None`, no label will be applied.

        warn_infer_x_label_with_multi_unit
            Should a warning be raised if we try to infer the x-unit
            but the data has more than one unit?

        y_label
            Label to apply to the y-axis.

            If `True`, we will try and infer the y-label based on the data's units.

            If `None`, no label will be applied.

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

        # TODO: split this back out
        # Logic applies to label, not unit label generation (?)
        # if isinstance(x_label, str) or x_label is None:
        #     # Don't add units to labels
        #     # TODO: consider giving option for units to be formatted
        #     # by some input function
        #     # (so user controls label, but units are injected)
        #     x_label_units = None
        #
        # if isinstance(x_label, bool) and not x_label:
        #     # No label to be generated, convert to None
        #     x_label_units = None

        if unit_var is None:
            # Nothing to generate from
            x_label_units = None

        if not isinstance(x_label, bool):
            msg = f"{type(x_label)} are not supported. {x_label=}"
            raise TypeError(msg)

        # No unit-aware plotting for scatter plots,
        # always want x_stacked_column and y_stacked_column in axis labels
        # by default.
        # # label is `True` from here on
        # if unit_aware:
        #     # Let unit-aware plotting do its thing
        #     x_label = None

        # Try to infer label
        if unit_var not in series.index.names:
            warnings.warn(
                "Not auto-generating the label "
                f"because {unit_var=} is not in {series.index.names=}",
                stacklevel=3,
            )
            return None

        values_units = series_x_relevant.index.get_level_values(unit_var)
        units_s = set(values_units)
        if len(units_s) == 1:
            x_label_units = values_units[0]
        else:
            # More than one unit plotted, don't infer a label
            if warn_infer_x_label_with_multi_unit:
                warnings.warn(
                    "Not auto-generating the label "
                    "because the data has more than one unit: "
                    f"data units {sorted(units_s)}",
                    stacklevel=3,
                )

            x_label_units = None

        if isinstance(x_label, str) or x_label is None:
            x_label = x_label
        elif isinstance(x_label, bool) and not x_label:
            x_label = None
        elif not isinstance(x_label, bool):
            msg = f"{type(x_label)} are not supported. {x_label=}"
            raise TypeError(msg)
        else:
            x_label = f"{x_stacked_column} [{x_label_units}]"

        # x_label_units = handle_axis_label_inference_from_unit_information(
        #     label=x_label,
        #     unit_aware=unit_aware,
        #     pandas_obj=series_x_relevant,
        #     unit_index_level=unit_var,
        #     warn_infer_label_with_multi_unit=warn_infer_x_label_with_multi_unit,
        # )

        y_label_units = handle_axis_label_inference_from_unit_information(
            label=y_label,
            unit_aware=unit_aware,
            pandas_obj=series_y_relevant,
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
            raise NotImplementedError
            style_items = [
                mlines.Line2D(
                    [],
                    [],
                    linestyle=linestyle,
                    label=style_value,
                    color="gray",
                )
                for style_value, linestyle in self.dashes.items()
            ]
            legend_items.append(mpatches.Patch(alpha=0, label=self.linestyle_var_label))
            legend_items.extend(style_items)

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
