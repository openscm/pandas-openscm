"""
Plotting
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable
from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
import pandas as pd
from attrs import define, field
from typing_extensions import TypeAlias

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)

if TYPE_CHECKING:
    import attr
    import matplotlib
    import pint

    COLOUR_VALUE_LIKE: TypeAlias = (
        str | tuple[float, float, float] | tuple[float, float, float, float]
    )
    """Types that allow a colour to be specified in matplotlib"""

    DASH_VALUE_LIKE: TypeAlias = str | tuple[float, tuple[float, ...]]
    """Types that allow a dash to be specified in matplotlib"""

    QUANTILES_PLUMES_LIKE: TypeAlias = tuple[
        tuple[float, float] | tuple[tuple[float, float], float], ...
    ]
    """Type that quantiles and the alpha to use for plotting their line/plume"""

    NP_ARRAY_OF_FLOAT_OR_INT = np.typing.NDArray[np.floating[Any] | np.integer[Any]]
    """Numpy array like"""

    PINT_NUMPY_ARRAY: TypeAlias = pint.facets.numpy.quantity.NumpyQuantity[
        NP_ARRAY_OF_FLOAT_OR_INT
    ]
    """
    Type alias for a pint quantity that wraps a numpy array

    No shape hints because that doesn't seem to be supported by numpy yet.
    """

    T = TypeVar("T")


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


def get_quantiles(quantiles_plumes: QUANTILES_PLUMES_LIKE) -> tuple[float, ...]:
    """
    Get just the quantiles from a `QUANTILES_PLUMES_LIKE`

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

    return tuple(set(quantiles_l))


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


def create_legend_default(
    ax: matplotlib.axes.Axes, handles: list[matplotlib.artist.Artist]
) -> None:
    """
    Create legend, default implementation

    Intended to be used with [plot_plume][(m).]

    Parameters
    ----------
    ax
        Axes on which to create the legend

    handles
        Handles to include in the legend
    """
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5))


def get_default_colour_cycler() -> Iterable[COLOUR_VALUE_LIKE]:
    """
    Get the default colour cycler

    Returns
    -------
    :
        Default colour cycler
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_default_colour_cycler", requirement="matplotlib"
        ) from exc

    colour_cycler = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    return colour_cycler


def fill_out_palette(
    hue_values: Iterable[T],
    palette_user_supplied: dict[T, COLOUR_VALUE_LIKE] | None,
    warn_on_value_missing: bool,
) -> dict[T, COLOUR_VALUE_LIKE]:
    if palette_user_supplied is None:
        # Make it all ourselves.
        # Don't warn as the user didn't set any values
        # so it is clear they want us to fill in everything.
        colour_cycler = get_default_colour_cycler()
        palette_out = {v: next(colour_cycler) for v in hue_values}

        return palette_out

    # User-supplied palette
    missing_from_user_supplied = [
        v for v in hue_values if v not in palette_user_supplied
    ]
    if not missing_from_user_supplied:
        # Just return the values we need
        return {v: palette_user_supplied[v] for v in hue_values}

    if warn_on_value_missing:
        msg = (
            f"Some hue values are not in the user-supplied palette, "
            "they will be filled from the default colour cycler instead. "
            f"{missing_from_user_supplied=} {palette_user_supplied=}"
        )
        warnings.warn(msg)

    palette_out = {}
    colour_cycler = get_default_colour_cycler()
    palette_out = {
        v: (
            palette_user_supplied[v]
            if v in palette_user_supplied
            else next(colour_cycler)
        )
        for v in hue_values
    }

    return palette_out


def get_default_dash_cycler() -> Iterable[DASH_VALUE_LIKE]:
    """
    Get the default dash cycler

    Returns
    -------
    :
        Default dash cycler
    """
    dash_cycler = cycle(["-", "--", "-.", ":"])

    return dash_cycler


def fill_out_dashes(
    style_values: Iterable[T],
    dashes_user_supplied: dict[T, DASH_VALUE_LIKE] | None,
    warn_on_value_missing: bool,
) -> dict[T, DASH_VALUE_LIKE]:
    if dashes_user_supplied is None:
        # Make it all ourselves.
        # Don't warn as the user didn't set any values
        # so it is clear they want us to fill in everything.
        dash_cycler = get_default_dash_cycler()
        dashes_out = {v: next(dash_cycler) for v in style_values}

        return dashes_out

    # User-supplied palette
    missing_from_user_supplied = [
        v for v in style_values if v not in dashes_user_supplied
    ]
    if not missing_from_user_supplied:
        # Just return the values we need
        return {v: dashes_user_supplied[v] for v in style_values}

    if warn_on_value_missing:
        msg = (
            f"{missing_from_user_supplied} not in the user-supplied palette, "
            "they will be filled from the default colour cycler instead. "
            f"{dashes_user_supplied=}"
        )
        warnings.warn(msg)

    dashes_out = {}
    dash_cycler = get_default_dash_cycler()
    for v in style_values:
        dashes_out[v] = (
            dashes_user_supplied[v] if v in dashes_user_supplied else next(dash_cycler)
        )

    return dashes_out


def y_vals_validator(
    obj: SingleLinePlotter | SinglePlumePlotter,
    attribute: attr.Attribute[Any],
    value: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY,
) -> None:
    """
    Validate the received y_vals
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
        validator=[y_vals_validator]
    )
    """y-values to plot"""

    quantile: float
    """Quantile that this line represents"""

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

    def get_label(self, quantile_legend_round: int = 2) -> str:
        """
        Get the label for the line

        Parameters
        ----------
        quantile_legend_round
            Rounding to apply to the quantile when creating the label

        Returns
        -------
        :
            Label for the line
        """
        label = np.round(self.quantile, quantile_legend_round)

        return label

    def plot(self, ax: matplotlib.axes.Axes, quantile_legend_round: int = 2) -> None:
        """
        Plot

        Parameters
        ----------
        ax
            Axes on which to plot

        quantile_legend_round
            Rounding to apply to the quantile when creating the label
        """
        pkwargs = self.pkwargs if self.pkwargs is not None else {}

        ax.plot(
            self.x_vals,
            self.y_vals,
            label=self.get_label(quantile_legend_round=quantile_legend_round),
            linewidth=self.linewidth,
            linestyle=self.linestyle,
            color=self.color,
            alpha=self.alpha,
            **pkwargs,
        )


@define
class SinglePlumePlotter:
    """Object which is able to plot single plumes"""

    x_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY
    """x-values to plot"""

    y_vals_lower: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY = field(
        validator=[y_vals_validator]
    )
    """y-values to plot as the lower bound of the plume"""

    y_vals_upper: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY = field(
        validator=[y_vals_validator]
    )
    """y-values to plot as the upper bound of the plume"""

    quantiles: tuple[float, float]
    """Quantiles that this plume represents"""

    color: COLOUR_VALUE_LIKE
    """Colour to use when plotting the plume"""

    alpha: float
    """Alpha to use when plotting the plume"""

    pkwargs: dict[str, Any] | None = None
    """Other arguments to pass to [matplotlib.axes.Axes.fill_between][] when plotting"""

    def get_label(self, quantile_legend_round: int = 2) -> str:
        """
        Get the label for the plume

        Parameters
        ----------
        quantile_legend_round
            Rounding to apply to the quantiles when creating the label

        Returns
        -------
        :
            Label for the plume
        """
        label = " - ".join(
            [str(np.round(qv, quantile_legend_round)) for qv in self.quantiles]
        )

        return label

    def plot(self, ax: matplotlib.axes.Axes, quantile_legend_round: int = 2) -> None:
        """
        Plot

        Parameters
        ----------
        ax
            Axes on which to plot

        quantile_legend_round
            Rounding to apply to the quantiles when creating the label
        """
        pkwargs = self.pkwargs if self.pkwargs is not None else {}

        ax.fill_between(
            self.x_vals,
            self.y_vals_lower,
            self.y_vals_upper,
            label=self.get_label(quantile_legend_round=quantile_legend_round),
            facecolor=self.color,
            alpha=self.alpha,
            **pkwargs,
        )


@define
class PlumePlotter:
    """Object which is able to plot plume plots"""

    lines: Iterable[SingleLinePlotter]
    """Lines to plot"""

    plumes: Iterable[SinglePlumePlotter]
    """Lines to plot"""

    hue_var_label: str
    """Label for the hue variable in the legend"""

    style_var_label: str | None
    """Label for the style variable in the legend (if not `None`)"""

    quantile_var_label: str
    """Label for the quantile variable in the legend"""

    palette: dict[Any, COLOUR_VALUE_LIKE]
    """Palette used for plotting different values of the hue variable"""

    dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None
    """Dashes used for plotting different values of the style variable"""

    x_label: str | None
    """Label to apply to the x-axis (if `None`, no label is applied)"""

    y_label: str | None
    """Label to apply to the y-axis (if `None`, no label is applied)"""

    @classmethod
    def from_df(  # noqa: PLR0912, PLR0913, PLR0915 # object creation code is the worst
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
        unit_col: str | None = "unit",
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

        Returns
        -------
        :
            Initialised instance
        """
        if hue_var_label is None:
            hue_var_label = hue_var.capitalize()

        if style_var is not None and style_var_label is None:
            style_var_label = style_var.capitalize()

        if quantile_var_label is None:
            quantile_var_label = quantile_var.capitalize()

        infer_y_label = isinstance(y_label, bool) and y_label and unit_col is not None

        palette_complete = fill_out_palette(
            df.index.get_level_values(hue_var).unique(),
            palette_user_supplied=palette,
            warn_on_value_missing=warn_on_palette_value_missing,
        )

        if style_var is not None:
            group_cols = [hue_var, style_var]
            dashes_complete = fill_out_dashes(
                df.index.get_level_values(style_var).unique(),
                dashes_user_supplied=dashes,
                warn_on_value_missing=warn_on_dashes_value_missing,
            )

        else:
            group_cols = [hue_var]
            dashes_complete = None

        lines: list[SingleLinePlotter] = []
        plumes: list[SinglePlumePlotter] = []
        values_units: list[str] = []
        for info, gdf in df.groupby(group_cols, observed=observed):
            info_d = {k: v for k, v in zip(group_cols, info)}

            colour = palette_complete[info_d[hue_var]]

            gpdf = partial(get_pdf_from_pre_calculated, gdf, quantile_col=quantile_var)

            def warn_about_missing_quantile(exc: Exception) -> pd.DataFrame | None:
                warnings.warn(
                    f"Quantiles missing for {info_d}. Original exception: {exc}"
                )

            for q, alpha in quantiles_plumes:
                if isinstance(q, float):
                    if style_var is not None:
                        linestyle = dashes_complete[info_d[style_var]]
                    else:
                        linestyle = "-"

                    try:
                        quantiles = (q,)
                        pdf = gpdf(quantiles=quantiles)
                    except MissingQuantileError as exc:
                        warn_about_missing_quantile(exc=exc)
                        continue

                    line_plotter = SingleLinePlotter(
                        x_vals=pdf.columns.values.squeeze(),
                        y_vals=pdf.values.squeeze(),
                        quantile=q,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        color=colour,
                        alpha=alpha,
                    )
                    lines.append(line_plotter)

                else:
                    try:
                        pdf = gpdf(quantiles=q)
                    except MissingQuantileError as exc:
                        warn_about_missing_quantile(exc=exc)
                        continue

                    plume_plotter = SinglePlumePlotter(
                        x_vals=pdf.columns.values.squeeze(),
                        y_vals_lower=gpdf(quantiles=(q[0],)).values.squeeze(),
                        y_vals_upper=gpdf(quantiles=(q[1],)).values.squeeze(),
                        quantiles=q,
                        color=colour,
                        alpha=alpha,
                    )
                    plumes.append(plume_plotter)

                if infer_y_label and unit_col in pdf.index.names:
                    values_units.extend(pdf.index.get_level_values(unit_col).unique())

        if infer_y_label:
            if unit_col not in df.index.names:
                warnings.warn(
                    "Not auto-setting the y_label "
                    f"because {unit_col=} is not in {df.index.names=}"
                )
                y_label = None

            else:
                # Try to infer the y-label
                units_s = set(values_units)
                if len(units_s) == 1:
                    y_label = values_units[0]
                else:
                    # More than one unit plotted, don't infer a y-label
                    if warn_infer_y_label_with_multi_unit:
                        warnings.warn(
                            "Not auto-setting the y_label "
                            "because the plotted data has more than one unit: "
                            f"data units {units_s}"
                        )

                    y_label = None

        res = PlumePlotter(
            lines=lines,
            plumes=plumes,
            hue_var_label=hue_var_label,
            style_var_label=style_var_label,
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

        generated_quantile_items = []
        quantile_items = []
        for line in self.lines:
            label = line.get_label(quantile_legend_round=quantile_legend_round)
            pid = (line.quantile, line.alpha, label)
            if pid in generated_quantile_items:
                continue

            quantile_items.append(
                mlines.Line2D([0], [0], color="k", alpha=line.alpha, label=label)
            )
            generated_quantile_items.append(pid)

        for plume in self.plumes:
            label = plume.get_label(quantile_legend_round=quantile_legend_round)
            pid = (plume.quantiles, plume.alpha, label)
            if pid in generated_quantile_items:
                continue

            quantile_items.append(
                mpatches.Patch(color="k", alpha=plume.alpha, label=label)
            )
            generated_quantile_items.append(pid)

        hue_items = [
            mlines.Line2D([0], [0], color=colour, label=hue_value)
            for hue_value, colour in self.palette.items()
        ]

        legend_items = [
            mpatches.Patch(alpha=0, label=self.quantile_var_label),
            *quantile_items,
            mpatches.Patch(alpha=0, label=self.hue_var_label),
            *hue_items,
        ]
        if self.dashes is not None and self.lines:
            style_items = [
                mlines.Line2D(
                    [0],
                    [0],
                    linestyle=linestyle,
                    label=style_value,
                    color="gray",
                )
                for style_value, linestyle in self.dashes.items()
            ]
            legend_items.append(mpatches.Patch(alpha=0, label=self.style_var_label))
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
            ax=ax,
            handles=self.generate_legend_handles(
                quantile_legend_round=quantile_legend_round
            ),
        )

        if self.x_label is not None:
            ax.set_xlabel(self.x_label)

        if self.y_label is not None:
            ax.set_ylabel(self.y_label)

        return ax


def plot_plume(  # noqa: PLR0913
    pdf: pd.DataFrame,
    ax: matplotlib.axes.Axes | None = None,
    *,
    quantiles_plumes: QUANTILES_PLUMES_LIKE = (
        (0.5, 0.7),
        ((0.05, 0.95), 0.2),
    ),
    quantile_var: str = "quantile",
    quantile_var_label: str | None = None,
    quantile_legend_round: int = 3,
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
    plotter = PlumePlotter.from_df(
        df=pdf,
        quantiles_plumes=quantiles_plumes,
        quantile_var=quantile_var,
        quantile_var_label=quantile_var_label,
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
        observed=observed,
    )

    plotter.plot(
        ax=ax, create_legend=create_legend, quantile_legend_round=quantile_legend_round
    )

    return ax


def plot_plume_after_calculating_quantiles(  # noqa: PLR0913
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
    quantile_var = "quantile"
    pdf_q = fix_index_name_after_groupby_quantile(
        groupby_except(pdf, quantile_over).quantile(get_quantiles(quantiles_plumes)),
        new_name=quantile_var,
        copy=False,
    )

    return plot_plume(
        pdf=pdf_q,
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
