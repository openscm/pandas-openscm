"""
Generation of line plots

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
    Protocol,
)

import pandas as pd
from attrs import define, field

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.plotting.axis_labels import (
    handle_axis_label_inference_from_unit_information,
)
from pandas_openscm.plotting.from_pandas_helpers import (
    fill_out_dashes,
    fill_out_palette,
    get_default_color_var_label,
    get_default_linestyle_var_label,
    get_values_line,
)
from pandas_openscm.plotting.legend import create_legend_default

if TYPE_CHECKING:
    import attr
    import matplotlib
    import pint

    from pandas_openscm.plotting.typing import (
        COLOUR_VALUE_LIKE,
        DASH_VALUE_LIKE,
        PALETTE_LIKE,
    )
    from pandas_openscm.typing import NP_ARRAY_OF_FLOAT_OR_INT, PINT_NUMPY_ARRAY


class HasXVals(Protocol):
    """Object that has x-values"""

    x_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY
    """x-values to plot"""


def is_same_shape_as_x_vals(
    obj: HasXVals,
    attribute: attr.Attribute[Any],
    value: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY,
) -> None:
    """
    Validate that the received values are the same shape as `obj.x_vals`

    Parameters
    ----------
    obj
        Object on which we are peforming validation

    attribute
        Attribute which is being set

    value
        Value which is being used to set `attribute`

    Raises
    ------
    AssertionError
        `value.shape` is not the same as `obj.x_vals.shape`
    """
    if value.shape != obj.x_vals.shape:
        msg = (
            f"`{attribute.name}` must have the same shape as `x_vals`. "
            f"Received `y_vals` with shape {value.shape} "
            f"while `x_vals` has shape {obj.x_vals.shape}"
        )
        raise AssertionError(msg)


@define
class SingleLinePlotter:
    """Object which is able to plot single lines"""

    x_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY
    """x-values to plot"""

    y_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY = field(
        validator=[is_same_shape_as_x_vals]
    )
    """y-values to plot"""

    linewidth: float
    """Linewidth to use when plotting the line"""

    linestyle: DASH_VALUE_LIKE
    """Style to use when plotting the line"""

    color: COLOUR_VALUE_LIKE
    """Colour to use when plotting the line"""

    alpha: float
    """Alpha to use when plotting the line"""

    pkwargs: dict[str, Any] | None = None
    """Other arguments to pass to [matplotlib.axes.Axes.plot][] when plotting"""

    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """
        Plot

        Parameters
        ----------
        ax
            Axes on which to plot
        """
        pkwargs = self.pkwargs if self.pkwargs is not None else {}

        ax.plot(
            self.x_vals,
            self.y_vals,
            linewidth=self.linewidth,
            linestyle=self.linestyle,
            color=self.color,
            alpha=self.alpha,
            **pkwargs,
        )

        return ax


@define
class SeabornLikeLinePlotter:
    """
    Seaborn-like plotter for line plots

    This is really just a data holder,
    which allows us to split the logic for preparing data
    from the logic of actually making plots.
    This is useful because we don't want all the individual lines
    to appear in the legend, rather only summaries of the hue and dash
    used for each line
    (achieving such behaviour with the available matplotlib API is difficult).

    If you use this class directly, be careful.
    It is easy to create inconsistencies between the lines to be plotted
    and the other information (which is used to create the legend entries).
    For example, if you alter `self.palette` without altering `self.lines` accordingly,
    you will get legend entries that don't correspond to any lines.

    It's 'seaborn-like' because it is based on similar ideas to
    [seaborn](https://seaborn.pydata.org/),
    but has been adjusted to suit the style of data we have.
    If you're looking at this, you may also want to consider
    raw seaborn (or even matplotlib), because those libraries
    provide much more flexibility than this API.
    """

    lines: Iterable[SingleLinePlotter]
    """Line plotters"""

    color_var_label: str
    """Label for the variable by which lines are coloured in the legend"""

    linestyle_var_label: str | None
    """Label for the variable by which lines are styled in the legend (if not `None`)"""

    palette: PALETTE_LIKE[Any]
    """
    Palette used for different values of the variable by which lines are coloured
    """

    dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None
    """
    Dashes used for different values of the variable by which lines are styled
    """

    x_label: str | None
    """Label to apply to the x-axis (if `None`, no label is applied)"""

    y_label: str | None
    """Label to apply to the y-axis (if `None`, no label is applied)"""

    @classmethod
    def from_df(  # noqa: PLR0913 # object creation code is the worst
        cls,
        df: pd.DataFrame,
        *,
        color_var: str = "scenario",
        color_var_label: str | None = None,
        palette: PALETTE_LIKE[Any] | None = None,
        warn_on_palette_value_missing: bool = True,
        linestyle_var: str | None = None,
        linestyle_var_label: str | None = None,
        dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None = None,
        warn_on_dashes_value_missing: bool = True,
        linewidth: float = 3.0,
        alpha: float = 0.8,
        unit_var: str | None = "unit",
        unit_aware: bool | pint.facets.PlainRegistry = False,
        time_units: str | None = None,
        x_label: str | None = "time",
        y_label: str | bool | None = True,
        warn_infer_y_label_with_multi_unit: bool = True,
        observed: bool = True,
    ) -> SeabornLikeLinePlotter:
        """
        Initialise from a [pd.DataFrame][pandas.DataFrame]

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame] from which to initialise

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

        linestyle_var
            Variable to use for grouping data into different linestyle groups

        linestyle_var_label
            Label to use as the header for the style section in the legend

        dashes
            Dash/linestyle to use for the different groups in the data.

            If any groups are not included in `dashes`,
            they are auto-filled.

        warn_on_dashes_value_missing
            Should a warning be emitted if there are values missing from `dashes`?

        linewidth
            Width to use for plotting lines.

        alpha
            Alpha to use when plotting the line

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

            If `None`, no label will be applied.

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

        if linestyle_var is not None and linestyle_var_label is None:
            linestyle_var_label = get_default_linestyle_var_label(linestyle_var)

        palette_complete = fill_out_palette(
            df,
            color_index_level=color_var,
            palette_user_supplied=palette,
            warn_on_value_missing=warn_on_palette_value_missing,
        )

        if linestyle_var is not None:
            group_cols = [color_var, linestyle_var]
            dashes_complete = fill_out_dashes(
                df,
                linestyle_index_level=linestyle_var,
                dashes_user_supplied=dashes,
                warn_on_value_missing=warn_on_dashes_value_missing,
            )

        else:
            group_cols = [color_var]
            dashes_complete = None

        lines: list[SingleLinePlotter] = []
        for info, gdf in df.groupby(group_cols, observed=observed):
            info_d = {k: v for k, v in zip(group_cols, info)}

            colour = palette_complete[info_d[color_var]]

            if linestyle_var is not None:
                if dashes_complete is None:  # pragma: no cover
                    # should be impossible to hit this
                    raise AssertionError
                linestyle = dashes_complete[info_d[linestyle_var]]

            else:
                linestyle = "-"

            line_plotter = SingleLinePlotter(
                *get_values_line(
                    gdf,
                    unit_aware=unit_aware,  # type: ignore # not sure why mypy is complaining
                    unit_var=unit_var,
                    time_units=time_units,
                ),
                linewidth=linewidth,
                linestyle=linestyle,
                color=colour,
                alpha=alpha,
            )
            lines.append(line_plotter)

        y_label = handle_axis_label_inference_from_unit_information(
            label=y_label,
            unit_aware=unit_aware,
            pandas_obj=df,
            unit_index_level=unit_var,
            warn_infer_label_with_multi_unit=warn_infer_y_label_with_multi_unit,
        )

        res = SeabornLikeLinePlotter(
            lines=lines,
            color_var_label=color_var_label,
            linestyle_var_label=linestyle_var_label,
            palette=palette_complete,
            dashes=dashes_complete,
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
        if self.dashes is not None:
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
        quantile_legend_round: int = 2,
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

        for line in self.lines:
            line.plot(ax=ax)

        create_legend(ax, self.generate_legend_handles())

        if self.x_label is not None:
            ax.set_xlabel(self.x_label)

        if self.y_label is not None:
            ax.set_ylabel(self.y_label)

        return ax
