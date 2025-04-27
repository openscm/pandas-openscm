"""
Plotting
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable, Iterator, Mapping
from functools import partial
from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
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

if TYPE_CHECKING:
    import attr
    import matplotlib
    import pint
    from typing_extensions import TypeAlias

    from pandas_openscm.typing import NP_ARRAY_OF_FLOAT_OR_INT, PINT_NUMPY_ARRAY

    COLOUR_VALUE_LIKE: TypeAlias = Union[
        Union[
            str,
            tuple[float, float, float],
            tuple[float, float, float, float],
            tuple[Union[tuple[float, float, float], str], float],
            tuple[tuple[float, float, float, float], float],
        ],
    ]
    """Type that allows a colour to be specified in matplotlib"""

    T = TypeVar("T")

    class PALETTE_LIKE(
        Generic[T],
        Mapping[T, COLOUR_VALUE_LIKE],
    ):
        """Palette-like type"""

    DASH_VALUE_LIKE: TypeAlias = Union[str, tuple[float, tuple[float, ...]]]
    """Types that allow a dash to be specified in matplotlib"""

    QUANTILES_PLUMES_LIKE: TypeAlias = tuple[
        Union[tuple[float, float], tuple[tuple[float, float], float]], ...
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


def extract_single_unit(df: pd.DataFrame, unit_var: str) -> str:
    """
    Extract the unit of the data, expecting there to only be one unit

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] from which to get the unit

    unit_var
        Variable/column in the multi-index which holds unit information

    Returns
    -------
    :
        Unit of the data

    Raises
    ------
    AssertionError
        The data has more than one unit
    """
    units = df.index.get_level_values(unit_var).unique().tolist()
    if len(units) != 1:
        raise AssertionError(units)

    return cast(str, units[0])


@overload
def get_values_line(
    pdf: pd.DataFrame,
    *,
    unit_aware: Literal[False],
    unit_var: str | None,
    time_units: str | None,
) -> tuple[NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT]: ...


@overload
def get_values_line(
    pdf: pd.DataFrame,
    *,
    unit_aware: Literal[True] | pint.facets.PlainRegistry,
    unit_var: str | None,
    time_units: str | None,
) -> tuple[PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY]: ...


def get_values_line(
    pdf: pd.DataFrame,
    *,
    unit_aware: bool | pint.facets.PlainRegistry,
    unit_var: str | None,
    time_units: str | None,
) -> (
    tuple[NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT]
    | tuple[PINT_NUMPY_ARRAY, PINT_NUMPY_ARRAY]
):
    """
    Get values for plotting a line

    Parameters
    ----------
    pdf
        [pd.DataFrame][pandas.DataFrame] from which to get the values

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

    y_values :
        y-values (for a plot)

    Raises
    ------
    TypeError
        `unit_aware` is not `False` and `unit_var` or `time_units` is `None`.

    MissingOptionalDependencyError
        `unit_aware` is `True`
        and [pint](https://pint.readthedocs.io/) is not installed.
    """
    res_no_units = (pdf.columns.values.squeeze(), pdf.values.squeeze())
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
                "get_values_line(..., unit_aware=True, ...)", requirement="pint"
            ) from exc

        ur = pint.get_application_registry()  # type: ignore

    else:
        ur = unit_aware

    res = (
        res_no_units[0] * ur(time_units),
        res_no_units[1] * ur(extract_single_unit(pdf, unit_var)),
    )

    return res


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


def create_legend_default(
    ax: matplotlib.axes.Axes, handles: list[matplotlib.artist.Artist]
) -> None:
    """
    Create legend, default implementation

    Intended to be used with [plot_plume_func][(m).]

    Parameters
    ----------
    ax
        Axes on which to create the legend

    handles
        Handles to include in the legend
    """
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5))


def get_default_colour_cycler() -> Iterator[COLOUR_VALUE_LIKE]:
    """
    Get the default colour cycler

    Returns
    -------
    :
        Default colour cycler

    Raises
    ------
    MissingOptionalDependencyError
        [matplotlib][] is not installed
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
    palette_user_supplied: PALETTE_LIKE[T] | None,
    warn_on_value_missing: bool,
) -> PALETTE_LIKE[T]:
    """
    Fill out a palette

    Parameters
    ----------
    hue_values
        Values which require a value in the output palette

    palette_user_supplied
        User-supplied palette

    warn_on_value_missing
        Should a warning be emitted if `palette_user_supplied` is not `None`
        but there are values missing from `palette_user_supplied`?

    Returns
    -------
    :
        Palette with values for all `hue_values`

    Warns
    -----
    UserWarning
        `warn_on_value_missing` is `True`,
        `palette_user_supplied` is not `None`
        and there are values in `hue_values` which are not in `palette_user_supplied`.
    """
    if palette_user_supplied is None:
        # Make it all ourselves.
        # Don't warn as the user didn't set any values
        # so it is clear they want us to fill in everything.
        colour_cycler = get_default_colour_cycler()
        palette_out: PALETTE_LIKE[T] = {  # type: ignore # not sure what I've done wrong
            v: next(colour_cycler) for v in hue_values
        }

        return palette_out

    # User-supplied palette
    missing_from_user_supplied = [
        v for v in hue_values if v not in palette_user_supplied
    ]
    if not missing_from_user_supplied:
        # Just return the values we need
        return {v: palette_user_supplied[v] for v in hue_values}  # type: ignore # not sure what mypy doesn't like

    if warn_on_value_missing:
        msg = (
            f"Some hue values are not in the user-supplied palette, "
            "they will be filled from the default colour cycler instead. "
            f"{missing_from_user_supplied=} {palette_user_supplied=}"
        )
        warnings.warn(msg)

    colour_cycler = get_default_colour_cycler()
    palette_out = {  # type: ignore # not sure what I've done wrong
        k: (
            palette_user_supplied[k]
            if k in palette_user_supplied
            else next(colour_cycler)
        )
        for k in hue_values
    }

    return palette_out


def get_default_dash_cycler() -> Iterator[DASH_VALUE_LIKE]:
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
    """
    Fill out dashes

    Parameters
    ----------
    style_values
        Values which require a value in the output dashes

    dashes_user_supplied
        User-supplied dashes

    warn_on_value_missing
        Should a warning be emitted if `dashes_user_supplied` is not `None`
        but there are values missing from `dashes_user_supplied`?

    Returns
    -------
    :
        Dashes with values for all `style_values`

    Warns
    -----
    UserWarning
        `warn_on_value_missing` is `True`,
        `dashes_user_supplied` is not `None`
        and there are values in `style_values` which are not in `dashes_user_supplied`.
    """
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
            f"Some style values are not in the user-supplied dashes, "
            "they will be filled from the default dash cycler instead. "
            f"{missing_from_user_supplied=} {dashes_user_supplied=}"
        )
        warnings.warn(msg)

    dashes_out = {}
    dash_cycler = get_default_dash_cycler()
    for v in style_values:
        dashes_out[v] = (
            dashes_user_supplied[v] if v in dashes_user_supplied else next(dash_cycler)
        )

    return dashes_out


def same_shape_as_x_vals(
    obj: SingleLinePlotter | SinglePlumePlotter,
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
        validator=[same_shape_as_x_vals]
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
        label = str(np.round(self.quantile, quantile_legend_round))

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
        validator=[same_shape_as_x_vals]
    )
    """y-values to plot as the lower bound of the plume"""

    y_vals_upper: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY = field(
        validator=[same_shape_as_x_vals]
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

    palette: PALETTE_LIKE[Any]
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
        palette: PALETTE_LIKE[Any] | None = None,
        warn_on_palette_value_missing: bool = True,
        style_var: str | None = "variable",
        style_var_label: str | None = None,
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

            If the first element of each tuple is a tuple,
            a [SinglePlumePlotter][(m).] object will be created.
            Otherwise, if the first element is a plain float,
            a [SingleLinePlotter][(m).] object will be created.

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
        if hue_var_label is None:
            hue_var_label = hue_var.capitalize()

        if style_var is not None and style_var_label is None:
            style_var_label = style_var.capitalize()

        if quantile_var_label is None:
            quantile_var_label = quantile_var.capitalize()

        infer_y_label = (
            (not unit_aware)
            and isinstance(y_label, bool)
            and y_label
            and unit_var is not None
        )

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

            def warn_about_missing_quantile(exc: Exception) -> None:
                warnings.warn(
                    f"Quantiles missing for {info_d}. Original exception: {exc}"
                )

            for q, alpha in quantiles_plumes:
                if isinstance(q, float):
                    if style_var is not None:
                        if dashes_complete is None:  # pragma: no cover
                            # should be impossible to hit this
                            raise AssertionError
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
                        *get_values_line(
                            pdf,
                            unit_aware=unit_aware,  # type: ignore # not sure why mypy is complaining
                            unit_var=unit_var,
                            time_units=time_units,
                        ),
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
                        *get_values_plume(
                            pdf,
                            quantiles=q,
                            quantile_var=quantile_var,
                            unit_aware=unit_aware,  # type: ignore # not sure why mypy is complaining
                            unit_var=unit_var,
                            time_units=time_units,
                        ),
                        quantiles=q,
                        color=colour,
                        alpha=alpha,
                    )
                    plumes.append(plume_plotter)

                if infer_y_label and unit_var in pdf.index.names:
                    values_units.extend(pdf.index.get_level_values(unit_var).unique())

        if unit_aware and isinstance(y_label, bool) and y_label:
            # Let unit-aware plotting do its thing
            y_label = None

        elif unit_var is None:
            y_label = None

        elif infer_y_label:
            if unit_var not in df.index.names:
                warnings.warn(
                    "Not auto-setting the y_label "
                    f"because {unit_var=} is not in {df.index.names=}"
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

        if isinstance(y_label, bool):
            msg = "y_label should have been converted before getting here"
            raise TypeError(msg)

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

        generated_quantile_items: list[
            Union[
                tuple[float, float, str],
                tuple[tuple[float, float], float, str],
            ]
        ] = []
        quantile_items: list[matplotlib.artist.Artist] = []
        for line in self.lines:
            label = line.get_label(quantile_legend_round=quantile_legend_round)
            pid_line = (line.quantile, line.alpha, label)
            if pid_line in generated_quantile_items:
                continue

            quantile_items.append(
                mlines.Line2D([0], [0], color="k", alpha=line.alpha, label=label)
            )
            generated_quantile_items.append(pid_line)

        for plume in self.plumes:
            label = plume.get_label(quantile_legend_round=quantile_legend_round)
            pid_plume = (plume.quantiles, plume.alpha, label)
            if pid_plume in generated_quantile_items:
                continue

            quantile_items.append(
                mpatches.Patch(color="k", alpha=plume.alpha, label=label)
            )
            generated_quantile_items.append(pid_plume)

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
