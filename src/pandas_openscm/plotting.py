"""
Plotting
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
from typing_extensions import TypeAlias

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)

if TYPE_CHECKING:
    import matplotlib

COLOUR_VALUE_LIKE: TypeAlias = (
    str | tuple[float, float, float] | tuple[float, float, float, float]
)
"""Types that allow a colour to be specified in matplotlib"""

QUANTILES_PLUMES_LIKE: TypeAlias = tuple[
    tuple[float, float] | tuple[tuple[float, float], float], ...
]
"""Type that quantiles and the alpha to use for plotting their line/plume"""


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
) -> pd.DataFrame | None:
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
    available_quantiles = in_df.index.get_level_values(quantile_col).unique()
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


def plot_plume(
    pdf: pd.DataFrame,
    ax: matplotlib.axes.Axes | None = None,
    *,
    quantile_over: str | list[str] = "run",
    quantiles_plumes: QUANTILES_PLUMES_LIKE = (
        (0.5, 0.7),
        ((0.05, 0.95), 0.2),
    ),
    hue_var: str = "scenario",
    hue_var_label: str | None = None,
    style_var: str = "variable",
    style_var_label: str | None = None,
    palette: dict[Any, COLOUR_VALUE_LIKE | tuple[COLOUR_VALUE_LIKE, float]]
    | None = None,
    dashes: dict[Any, str | tuple[float, tuple[float, ...]]] | None = None,
    linewidth: float = 3.0,
    unit_col: str = "unit",
    pre_calculated: bool = False,
    quantile_col: str = "quantile",
    quantile_col_label: str | None = None,
    observed: bool = True,
    y_label: str | bool | None = True,
    x_label: str | None = "time",
    create_legend: Callable[
        [matplotlib.axes.Axes, list[matplotlib.artist.Artist]], None
    ] = create_legend_default,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plot_plume", requirement="matplotlib"
        ) from exc

    if ax is None:
        _, ax = plt.subplots()

    # The joy of plotting, you create everything yourself.
    if hue_var_label is None:
        hue_var_label = hue_var.capitalize()
    if style_var_label is None:
        style_var_label = style_var.capitalize()
    if quantile_col_label is None:
        quantile_col_label = quantile_col.capitalize()

    quantiles = get_quantiles(quantiles_plumes)

    _palette = {} if palette is None else palette

    if dashes is None:
        _dashes = {}
        lines = ["-", "--", "-.", ":"]
        linestyle_cycler = cycle(lines)
    else:
        _dashes = dashes

    if create_legend is None:
        create_legend = create_legend_default

    # Need to keep track of this, just in case we end up plotting only plumes
    _plotted_lines = False

    quantile_labels = {}
    plotted_hues = []
    plotted_styles = []
    units_l = []
    for q, alpha in quantiles_plumes:
        if isinstance(q, float):
            quantiles = (q,)
            plot_plume = False
        else:
            quantiles = q
            plot_plume = True

        # Can split out plot plume vs. plot line to use here

        for hue_value, hue_ts in pdf.groupby(hue_var, observed=observed):
            for style_value, hue_style_ts in hue_ts.groupby(
                style_var, observed=observed
            ):
                # Remake in inner loop to avoid leaking between plots
                pkwargs = {"alpha": alpha}

                if pre_calculated:
                    try:
                        pdf_group = get_pdf_from_pre_calculated(
                            hue_ts,
                            quantiles=quantiles,
                            quantile_col=quantile_col,
                        )

                    except MissingQuantileError as exc:
                        warnings.warn(
                            f"Missing quantiles for {hue_value=} {style_value=}. "
                            f"Original exception: {exc}"
                        )
                        continue

                else:
                    tmp = groupby_except(hue_ts, quantile_over).quantile(quantiles)
                    pdf_group = fix_index_name_after_groupby_quantile(
                        tmp, new_name=quantile_col
                    )

                if hue_value not in plotted_hues:
                    plotted_hues.append(hue_value)

                x_vals = pdf_group.columns.values.squeeze()
                # Require ur for this to work
                # if time_units is not None:
                #     x_vals = x_vals * time_units
                # x_vals = get_plot_vals(
                #     x_vals,
                #     "x_axis",
                #     warn_if_magnitudes=warn_if_plotting_magnitudes,
                # )

                if palette is not None:
                    try:
                        pkwargs["color"] = palette[hue_value]
                    except KeyError:
                        error_msg = f"{hue_value} not in palette. {palette=}"
                        raise KeyError(error_msg)

                elif hue_value in _palette:
                    pkwargs["color"] = _palette[hue_value]
                # else:
                #     # Let matplotlib default cycling do its thing

                if plot_plume:
                    label = f"{q[0] * 100:.0f}th - {q[1] * 100:.0f}th"

                    y_lower_vals = pdf_group.loc[
                        pdf_group.index.get_level_values(quantile_col).isin({q[0]})
                    ].values.squeeze()
                    y_upper_vals = pdf_group.loc[
                        pdf_group.index.get_level_values(quantile_col).isin({q[1]})
                    ].values.squeeze()
                    # Require ur for this to work
                    # Also need the 1D check back in too
                    # y_lower_vals = get_plot_vals(
                    #     y_lower_vals * y_units,
                    #     "y_lower_vals",
                    #     warn_if_magnitudes=warn_if_plotting_magnitudes,
                    # )
                    p = ax.fill_between(
                        x_vals,
                        y_lower_vals,
                        y_upper_vals,
                        label=label,
                        **pkwargs,
                    )

                    if palette is None:
                        _palette[hue_value] = p.get_facecolor()[0]

                else:
                    if style_value not in plotted_styles:
                        plotted_styles.append(style_value)

                    _plotted_lines = True

                    if dashes is not None:
                        try:
                            pkwargs["linestyle"] = _dashes[style_value]
                        except KeyError:
                            error_msg = f"{style_value} not in dashes. {dashes=}"
                            raise KeyError(error_msg)
                    else:
                        if style_value not in _dashes:
                            _dashes[style_value] = next(linestyle_cycler)

                        pkwargs["linestyle"] = _dashes[style_value]

                    if isinstance(q, str):
                        label = q
                    else:
                        label = f"{q * 100:.0f}th"

                    y_vals = pdf_group.loc[
                        pdf_group.index.get_level_values(quantile_col).isin({q})
                    ].values.squeeze()
                    # Require ur for this to work
                    # Also need the 1D check back in too
                    # y_lower_vals = get_plot_vals(
                    #     y_lower_vals * y_units,
                    #     "y_lower_vals",
                    #     warn_if_magnitudes=warn_if_plotting_magnitudes,
                    # )
                    p = ax.plot(
                        x_vals,
                        y_vals,
                        label=label,
                        linewidth=linewidth,
                        **pkwargs,
                    )[0]

                    if dashes is None:
                        _dashes[style_value] = p.get_linestyle()

                    if palette is None:
                        _palette[hue_value] = p.get_color()

                if label not in quantile_labels:
                    quantile_labels[label] = p

                # Once we have unit handling with matplotlib, we can remove this
                # (and if matplotlib isn't set up, we just don't do unit handling)
                # TODO: if clause here to check re unit handling
                units_l.extend(
                    pdf_group.index.get_level_values(unit_col).unique().tolist()
                )

    # Fake the line handles for the legend
    hue_val_lines = [
        mlines.Line2D([0], [0], color=_palette[hue_value], label=hue_value)
        for hue_value in plotted_hues
    ]

    legend_items = [
        mpatches.Patch(alpha=0, label=quantile_col_label),
        *quantile_labels.values(),
        mpatches.Patch(alpha=0, label=hue_var_label),
        *hue_val_lines,
    ]

    if _plotted_lines:
        style_val_lines = [
            mlines.Line2D(
                [0],
                [0],
                linestyle=_dashes[style_value],
                label=style_value,
                color="gray",
                linewidth=linewidth,
            )
            for style_value in plotted_styles
        ]
        legend_items += [
            mpatches.Patch(alpha=0, label=style_var_label),
            *style_val_lines,
        ]

    elif dashes is not None:
        warnings.warn(
            "`dashes` was passed but no lines were plotted, the style settings "
            "will not be used"
        )

    create_legend(ax=ax, handles=legend_items)

    if isinstance(y_label, bool) and y_label:
        if len(set(units_l)) == 1:
            ax.set_ylabel(units_l[0])
        else:
            warnings.warn(
                "Not auto-setting the y_label "
                f"because the plotted data has more than one unit: data units {units_l}"
            )

    elif y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:  # and units not already handled
        ax.set_xlabel(x_label)

    return ax, legend_items
