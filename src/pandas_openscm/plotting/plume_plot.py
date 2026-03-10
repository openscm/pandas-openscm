"""
Generation of plume plots

These allow the user to specify quantiles, for which plumes are plotted.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from attrs import define, field

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.plotting.axis_labels import (
    handle_axis_label_inference_from_unit_information,
)
from pandas_openscm.plotting.from_pandas_helpers import (
    extract_single_unit,
    fill_out_dashes,
    fill_out_palette,
    get_default_color_var_label,
    get_default_linestyle_var_label,
    get_default_quantile_var_label,
    get_values_line,
)
from pandas_openscm.plotting.legend import create_legend_default
from pandas_openscm.plotting.line_plot import SingleLinePlotter, is_same_shape_as_x_vals

if TYPE_CHECKING:
    import matplotlib
    import pint

    from pandas_openscm.plotting.typing import (
        COLOUR_VALUE_LIKE,
        PALETTE_LIKE,
        QUANTILES_PLUMES_LIKE,
    )
    from pandas_openscm.typing import NP_ARRAY_OF_FLOAT_OR_INT, PINT_NUMPY_ARRAY


class MissingQuantileError(KeyError):
    """
    Raised when a quantile(s) is missing from a [pd.DataFrame][pandas.DataFrame]
    """

    def __init__(
        self,
        available_quantiles: Collection[float],
        missing_quantiles: Collection[float],
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        available_quantiles
            Available quantiles

        missing_quantiles
            Missing quantiles
        """
        error_msg = (
            f"The folllowing quantiles are missing: {missing_quantiles=}. "
            f"{available_quantiles=}"
        )
        super().__init__(error_msg)


def get_quantiles(
    quantiles_plumes: QUANTILES_PLUMES_LIKE,
) -> np.typing.NDArray[np.floating[Any]]:
    """
    Get just the quantiles from a [QUANTILES_PLUMES_LIKE][(m).]

    Parameters
    ----------
    quantiles_plumes
        Quantiles-plumes definition

    Returns
    -------
    :
        Quantiles to be used in plotting
    """
    quantiles_l = []
    for quantile_plot_def in quantiles_plumes:
        q_def = quantile_plot_def[0]
        if isinstance(q_def, float):
            quantiles_l.append(q_def)
        else:
            for q in q_def:
                quantiles_l.append(q)

    return np.unique(np.array(quantiles_l))  # type: ignore # numpy and mypy not playing nice


def get_pdf_from_pre_calculated(
    in_df: pd.DataFrame,
    *,
    quantiles: Iterable[float],
    quantile_col: str,
) -> pd.DataFrame:
    """
    Get a [pd.DataFrame][pandas.DataFrame] for plotting from pre-calculated quantiles

    Parameters
    ----------
    in_df
        Input [pd.DataFrame][pandas.DataFrame]

    quantiles
        Quantiles to grab

    quantile_col
        Name of the index column in which quantile information is stored

    Returns
    -------
    :
        [pd.DataFrame][pandas.DataFrame] to use for plotting.

    Raises
    ------
    MissingQuantileError
        One of the quantiles in `quantiles` is not available in `in_df`.
    """
    missing_quantiles = []
    available_quantiles = in_df.index.get_level_values(quantile_col).unique().tolist()
    for qt in quantiles:
        if qt not in available_quantiles:
            missing_quantiles.append(qt)

    if missing_quantiles:
        raise MissingQuantileError(available_quantiles, missing_quantiles)

    # otherwise, have what we need
    pdf = in_df.loc[in_df.index.get_level_values(quantile_col).isin(quantiles)]

    return pdf


@overload
def get_values_plume(
    pdf: pd.DataFrame,
    *,
    quantiles: tuple[float, float],
    quantile_var: str,
    unit_aware: Literal[False],
    unit_var: str | None,
    time_units: str | None,
) -> tuple[
    NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT
]: ...


@overload
def get_values_plume(
    pdf: pd.DataFrame,
    *,
    quantiles: tuple[float, float],
    quantile_var: str,
    unit_aware: Literal[True] | pint.facets.PlainRegistry,
    unit_var: str | None,
    time_units: str | None,
) -> tuple[PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY]: ...


def get_values_plume(  # noqa: PLR0913
    pdf: pd.DataFrame,
    *,
    quantiles: tuple[float, float],
    quantile_var: str,
    unit_aware: bool | pint.facets.PlainRegistry,
    unit_var: str | None,
    time_units: str | None,
) -> (
    tuple[NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT]
    | tuple[PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY]
):
    """
    Get values for plotting a line

    Parameters
    ----------
    pdf
        [pd.DataFrame][pandas.DataFrame] from which to get the values

    quantiles
        Quantiles to get from `pdf`

    quantile_var
        Variable/column in the multi-index which stores information
        about the quantile that each timeseries represents.

    unit_aware
        Should the values be unit-aware?

        If `True`, we use the default application registry
        (retrieved with [pint.get_application_registry][]).
        Otherwise, a [pint.facets.PlainRegistry][] can be supplied and will be used.

    unit_var
        Variable/column in the multi-index which stores information
        about the unit of each timeseries.

    time_units
        Units of the time axis.

    Returns
    -------
    x_values :
        x-values (for a plot)

    y_values_lower :
        y-values for the lower-bound (of a plume plot)

    y_values_upper :
        y-values for the upper-bound (of a plume plot)

    Raises
    ------
    TypeError
        `unit_aware` is not `False` and `unit_var` or `time_units` is `None`.

    MissingOptionalDependencyError
        `unit_aware` is `True`
        and [pint](https://pint.readthedocs.io/) is not installed.
    """
    res_no_units = (
        pdf.columns.values.squeeze(),
        pdf.loc[
            pdf.index.get_level_values(quantile_var).isin({quantiles[0]})
        ].values.squeeze(),
        pdf.loc[
            pdf.index.get_level_values(quantile_var).isin({quantiles[1]})
        ].values.squeeze(),
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
                "get_values_plume(..., unit_aware=True, ...)", requirement="pint"
            ) from exc

        ur = pint.get_application_registry()  # type: ignore

    else:
        ur = unit_aware

    unit = extract_single_unit(pdf, unit_var)
    res = (
        res_no_units[0] * ur(time_units),
        res_no_units[1] * ur(unit),
        res_no_units[2] * ur(unit),
    )

    return res


@define
class QuantileLinePlotter:
    """Object which plots single lines representing quantile"""

    quantile: float
    """Quantile that this line represents"""

    single_line_plotter: SingleLinePlotter
    """Plotter for the line"""

    def get_label_quantile(self, round: int = 2) -> str:
        """
        Get the label for the line's quantile information

        Parameters
        ----------
        round
            Rounding to apply to the quantile when creating the label

        Returns
        -------
        :
            Label for the line
        """
        label = str(np.round(self.quantile, round))

        return label

    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """
        Plot

        Parameters
        ----------
        ax
            Axes on which to plot
        """
        return self.single_line_plotter.plot(ax=ax)


@define
class SinglePlumePlotter:
    """Object which is able to plot single plumes"""

    x_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY
    """x-values to plot"""

    y_vals_lower: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY = field(
        validator=[is_same_shape_as_x_vals]
    )
    """y-values to plot as the lower bound of the plume"""

    y_vals_upper: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY = field(
        validator=[is_same_shape_as_x_vals]
    )
    """y-values to plot as the upper bound of the plume"""

    color: COLOUR_VALUE_LIKE
    """Colour to use when plotting the plume"""

    alpha: float
    """Alpha to use when plotting the plume"""

    pkwargs: dict[str, Any] | None = None
    """Other arguments to pass to [matplotlib.axes.Axes.fill_between][] when plotting"""

    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """
        Plot

        Parameters
        ----------
        ax
            Axes on which to plot
        """
        pkwargs = self.pkwargs if self.pkwargs is not None else {}

        ax.fill_between(
            self.x_vals,
            self.y_vals_lower,
            self.y_vals_upper,
            facecolor=self.color,
            alpha=self.alpha,
            **pkwargs,
        )

        return ax


@define
class QuantilePlumePlotter:
    """Object which plots single plumes representing a range between two quantiles"""

    quantiles: tuple[float, float]
    """Quantiles that this plume represents"""

    single_plume_plotter: SinglePlumePlotter
    """Plotter for the plume"""

    def get_label_quantile(self, round: int = 2) -> str:
        """
        Get the label for the plume's quantile information

        Parameters
        ----------
        round
            Rounding to apply to the quantiles when creating the label

        Returns
        -------
        :
            Label for the plume
        """
        label = " - ".join([str(np.round(qv, round)) for qv in self.quantiles])

        return label

    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """
        Plot

        Parameters
        ----------
        ax
            Axes on which to plot
        """
        return self.single_plume_plotter.plot(ax=ax)


@define
class PlumePlotter:
    """
    Object which is able to plot (quantile) plume plots
    """

    lines: Iterable[QuantileLinePlotter]
    """Line plotters"""

    plumes: Iterable[QuantilePlumePlotter]
    """Plume plotters"""

    color_var_label: str
    """Label for the colour variable in the legend"""

    linestyle_var_label: str | None
    """Label for the linestyle variable in the legend (if not `None`)"""

    quantile_var_label: str
    """Label for the quantile variable in the legend"""

    palette: PALETTE_LIKE[Any]
    """Palette used for plotting different values of the colour variable"""

    dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None
    """Dashes used for plotting different values of the style variable"""

    x_label: str | None
    """Label to apply to the x-axis (if `None`, no label is applied)"""

    y_label: str | None
    """Label to apply to the y-axis (if `None`, no label is applied)"""

    @classmethod
    def from_df(  # noqa: PLR0912, PLR0913 # object creation code is the worst
        cls,
        df: pd.DataFrame,
        *,
        quantiles_plumes: QUANTILES_PLUMES_LIKE = (
            (0.5, 0.7),
            ((0.05, 0.95), 0.2),
        ),
        quantile_var: str = "quantile",
        quantile_var_label: str | None = None,
        quantile_legend_round: int = 2,
        color_var: str = "scenario",
        color_var_label: str | None = None,
        palette: PALETTE_LIKE[Any] | None = None,
        warn_on_palette_value_missing: bool = True,
        linestyle_var: str | None = "variable",
        linestyle_var_label: str | None = None,
        dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None = None,
        warn_on_dashes_value_missing: bool = True,
        linewidth: float = 3.0,
        unit_var: str | None = "unit",
        unit_aware: bool | pint.facets.PlainRegistry = False,
        time_units: str | None = None,
        x_label: str | None = "time",
        y_label: str | bool | None = True,
        warn_infer_y_label_with_multi_unit: bool = True,
        observed: bool = True,
    ) -> PlumePlotter:
        """
        Initialise from a [pd.DataFrame][pandas.DataFrame]

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame] from which to initialise

        quantiles_plumes
            Quantiles to plot in each plume.

            If the first sub-element of each element is a plain float,
            a [SingleLinePlotter][(m).] object will be created.
            Otherwise, a [SinglePlumePlotter][(m).] object will be created
            on the assumption that the first sub-element is itself a tuple
            with two elements (the quantile range to show for this plume).

        quantile_var
            Variable/column in the multi-index which stores information
            about the quantile that each timeseries represents.

        quantile_var_label
            Label to use as the header for the quantile section in the legend

        quantile_legend_round
            Rounding to apply to quantile values when creating the legend

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
            Label to use as the header for the linestyle section in the legend

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

        if quantile_var_label is None:
            quantile_var_label = get_default_quantile_var_label(quantile_var)

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

        lines: list[QuantileLinePlotter] = []
        plumes: list[QuantilePlumePlotter] = []
        for info, gdf in df.groupby(group_cols, observed=observed):
            info_d = {k: v for k, v in zip(group_cols, info)}

            colour = palette_complete[info_d[color_var]]

            gpdf = partial(get_pdf_from_pre_calculated, gdf, quantile_col=quantile_var)

            def warn_about_missing_quantile(exc: Exception) -> None:
                warnings.warn(
                    f"Quantiles missing for {info_d}. Original exception: {exc}"
                )

            for q, alpha in quantiles_plumes:
                if isinstance(q, float):
                    if linestyle_var is not None:
                        if dashes_complete is None:  # pragma: no cover
                            # should be impossible to hit this
                            raise AssertionError
                        linestyle = dashes_complete[info_d[linestyle_var]]
                    else:
                        linestyle = "-"

                    try:
                        quantiles = (q,)
                        pdf = gpdf(quantiles=quantiles)
                    except MissingQuantileError as exc:
                        warn_about_missing_quantile(exc=exc)
                        continue

                    line_plotter = QuantileLinePlotter(
                        quantile=q,
                        single_line_plotter=SingleLinePlotter(
                            *get_values_line(
                                pdf,
                                unit_aware=unit_aware,  # type: ignore # not sure why mypy is complaining
                                unit_var=unit_var,
                                time_units=time_units,
                            ),
                            linewidth=linewidth,
                            linestyle=linestyle,
                            color=colour,
                            alpha=alpha,
                        ),
                    )
                    lines.append(line_plotter)

                else:
                    try:
                        pdf = gpdf(quantiles=q)
                    except MissingQuantileError as exc:
                        warn_about_missing_quantile(exc=exc)
                        continue

                    plume_plotter = QuantilePlumePlotter(
                        quantiles=q,
                        single_plume_plotter=SinglePlumePlotter(
                            *get_values_plume(
                                pdf,
                                quantiles=q,
                                quantile_var=quantile_var,
                                unit_aware=unit_aware,  # type: ignore # not sure why mypy is complaining
                                unit_var=unit_var,
                                time_units=time_units,
                            ),
                            color=colour,
                            alpha=alpha,
                        ),
                    )
                    plumes.append(plume_plotter)

        y_label = handle_axis_label_inference_from_unit_information(
            label=y_label,
            unit_aware=unit_aware,
            pandas_obj=df,
            unit_index_level=unit_var,
            warn_infer_label_with_multi_unit=warn_infer_y_label_with_multi_unit,
        )

        res = PlumePlotter(
            lines=lines,
            plumes=plumes,
            color_var_label=color_var_label,
            linestyle_var_label=linestyle_var_label,
            quantile_var_label=quantile_var_label,
            palette=palette_complete,
            dashes=dashes_complete,
            x_label=x_label,
            y_label=y_label,
        )

        return res

    def generate_legend_handles(
        self, quantile_legend_round: int = 2
    ) -> list[matplotlib.artist.Artist]:
        """
        Generate handles for the legend

        Parameters
        ----------
        quantile_legend_round
            Rounding to apply to the quantiles when creating the label

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

        generated_quantile_items: list[
            Union[
                tuple[float, float, str],
                tuple[tuple[float, float], float, str],
            ]
        ] = []
        quantile_items: list[matplotlib.artist.Artist] = []
        for line in self.lines:
            label = line.get_label_quantile(round=quantile_legend_round)
            pid_line = (line.quantile, line.single_line_plotter.alpha, label)
            if pid_line in generated_quantile_items:
                continue

            quantile_items.append(
                mlines.Line2D(
                    [],
                    [],
                    color="k",
                    alpha=line.single_line_plotter.alpha,
                    label=label,
                )
            )
            generated_quantile_items.append(pid_line)

        for plume in self.plumes:
            label = plume.get_label_quantile(round=quantile_legend_round)
            pid_plume = (plume.quantiles, plume.single_plume_plotter.alpha, label)
            if pid_plume in generated_quantile_items:
                continue

            quantile_items.append(
                mpatches.Patch(
                    color="k", alpha=plume.single_plume_plotter.alpha, label=label
                )
            )
            generated_quantile_items.append(pid_plume)

        color_items = [
            mlines.Line2D([], [], color=color, label=color_value)
            for color_value, color in self.palette.items()
        ]

        legend_items = [
            mpatches.Patch(alpha=0, label=self.quantile_var_label),
            *quantile_items,
            mpatches.Patch(alpha=0, label=self.color_var_label),
            *color_items,
        ]
        if self.dashes is not None and self.lines:
            linestyle_items = [
                mlines.Line2D(
                    [],
                    [],
                    linestyle=linestyle,
                    label=linestyle_value,
                    color="gray",
                )
                for linestyle_value, linestyle in self.dashes.items()
            ]
            legend_items.append(mpatches.Patch(alpha=0, label=self.linestyle_var_label))
            legend_items.extend(linestyle_items)

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

        quantile_legend_round
            Rounding to apply to quantile values when creating the legend

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

        for plume in self.plumes:
            plume.plot(ax=ax)

        for line in self.lines:
            line.plot(ax=ax)

        create_legend(
            ax,
            self.generate_legend_handles(quantile_legend_round=quantile_legend_round),
        )

        if self.x_label is not None:
            ax.set_xlabel(self.x_label)

        if self.y_label is not None:
            ax.set_ylabel(self.y_label)

        return ax


# Something funny happening with relative x-refs, hence _func suffix
def plot_plume_func(  # noqa: PLR0913
    pdf: pd.DataFrame,
    quantiles_plumes: QUANTILES_PLUMES_LIKE,
    ax: matplotlib.axes.Axes | None = None,
    *,
    quantile_var: str = "quantile",
    quantile_var_label: str | None = None,
    quantile_legend_round: int = 3,
    color_var: str = "scenario",
    color_var_label: str | None = None,
    palette: PALETTE_LIKE[Any] | None = None,
    warn_on_palette_value_missing: bool = True,
    linestyle_var: str = "variable",
    linestyle_var_label: str | None = None,
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
    pdf
        [pd.DataFrame][pandas.DataFrame] to use for plotting

        It must contain quantiles already.
        For data without quantiles, please see
        [plot_plume_after_calculating_quantiles_func][(m).].

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
        Label to use as the header for the linestyle section in the legend

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
    plotter = PlumePlotter.from_df(
        df=pdf,
        quantiles_plumes=quantiles_plumes,
        quantile_var=quantile_var,
        quantile_var_label=quantile_var_label,
        color_var=color_var,
        color_var_label=color_var_label,
        palette=palette,
        warn_on_palette_value_missing=warn_on_palette_value_missing,
        linestyle_var=linestyle_var,
        linestyle_var_label=linestyle_var_label,
        dashes=dashes,
        warn_on_dashes_value_missing=warn_on_dashes_value_missing,
        linewidth=linewidth,
        unit_var=unit_var,
        unit_aware=unit_aware,
        time_units=time_units,
        x_label=x_label,
        y_label=y_label,
        warn_infer_y_label_with_multi_unit=warn_infer_y_label_with_multi_unit,
        observed=observed,
    )

    ax = plotter.plot(
        ax=ax, create_legend=create_legend, quantile_legend_round=quantile_legend_round
    )

    return ax


# Something funny happening with relative x-refs, hence _func suffix
def plot_plume_after_calculating_quantiles_func(  # noqa: PLR0913
    pdf: pd.DataFrame,
    ax: matplotlib.axes.Axes | None = None,
    *,
    quantile_over: str | list[str],
    quantiles_plumes: QUANTILES_PLUMES_LIKE = (
        (0.5, 0.7),
        ((0.05, 0.95), 0.2),
    ),
    quantile_var_label: str | None = None,
    quantile_legend_round: int = 2,
    color_var: str = "scenario",
    color_var_label: str | None = None,
    palette: PALETTE_LIKE[Any] | None = None,
    warn_on_palette_value_missing: bool = True,
    linestyle_var: str = "variable",
    linestyle_var_label: str | None = None,
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
    pdf
        [pd.DataFrame][pandas.DataFrame] to use for plotting

        It must contain quantiles already.
        For data without quantiles, please see
        [plot_plume_after_calculating_quantiles_func][(m).].

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
        Label to use as the header for the linestyle section in the legend

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
    quantile_var = "quantile"
    pdf_q = fix_index_name_after_groupby_quantile(
        groupby_except(pdf, quantile_over).quantile(get_quantiles(quantiles_plumes)),
        new_name=quantile_var,
        copy=False,
    )

    return plot_plume_func(
        pdf=pdf_q,
        ax=ax,
        quantiles_plumes=quantiles_plumes,
        quantile_var=quantile_var,
        quantile_var_label=quantile_var_label,
        quantile_legend_round=quantile_legend_round,
        color_var=color_var,
        color_var_label=color_var_label,
        palette=palette,
        warn_on_palette_value_missing=warn_on_palette_value_missing,
        linestyle_var=linestyle_var,
        linestyle_var_label=linestyle_var_label,
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
