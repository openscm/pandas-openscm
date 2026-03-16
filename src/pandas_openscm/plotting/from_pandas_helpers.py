"""
Helpers for initialisation of plotting objects from [pandas][] objects
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
    overload,
)

from pandas_openscm.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import pandas as pd
    import pint

    from pandas_openscm.plotting.typing import (
        COLOUR_VALUE_LIKE,
        DASH_VALUE_LIKE,
        MARKER_VALUE_LIKE,
        PALETTE_LIKE,
    )
    from pandas_openscm.typing import NP_ARRAY_OF_FLOAT_OR_INT, PINT_NUMPY_ARRAY

    T = TypeVar("T")
    P = TypeVar("P", pd.DataFrame, pd.Series[Any])


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
    pandas_obj: P,
    linestyle_index_level: str,
    dashes_user_supplied: dict[T, DASH_VALUE_LIKE] | None,
    warn_on_value_missing: bool,
) -> dict[T, DASH_VALUE_LIKE]:
    """
    Fill out dashes

    Parameters
    ----------
    pandas_obj
        Pandas object for which to fill out the dashes

    linestyle_index_level
        Index level in `pandas_obj` from which to get the values
        which require a value in the output dashes

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
    linestyle_values = pandas_obj.index.get_level_values(linestyle_index_level).unique()

    if dashes_user_supplied is None:
        # Make it all ourselves.
        # Don't warn as the user didn't set any values
        # so it is clear they want us to fill in everything.
        dash_cycler = get_default_dash_cycler()
        dashes_out = {v: next(dash_cycler) for v in linestyle_values}

        return dashes_out

    # User-supplied palette
    missing_from_user_supplied = [
        v for v in linestyle_values if v not in dashes_user_supplied
    ]
    if not missing_from_user_supplied:
        # Just return the values we need
        return {v: dashes_user_supplied[v] for v in linestyle_values}

    if warn_on_value_missing:
        msg = (
            f"Some style values are not in the user-supplied dashes, "
            "they will be filled from the default dash cycler instead. "
            f"{missing_from_user_supplied=} {dashes_user_supplied=}"
        )
        warnings.warn(msg)

    dashes_out = {}
    dash_cycler = get_default_dash_cycler()
    for v in linestyle_values:
        dashes_out[v] = (
            dashes_user_supplied[v] if v in dashes_user_supplied else next(dash_cycler)
        )

    return dashes_out


def fill_out_markers(
    pandas_obj: P,
    marker_index_level: str,
    markers_user_supplied: dict[T, MARKER_VALUE_LIKE] | None,
    warn_on_value_missing: bool,
) -> dict[T, MARKER_VALUE_LIKE]:
    """
    Fill out markers

    Parameters
    ----------
    pandas_obj
        Pandas object for which to fill out the markers

    marker_index_level
        Index level in `pandas_obj` from which to get the values
        which require a value in the output markers

    markers_user_supplied
        User-supplied markers

    warn_on_value_missing
        Should a warning be emitted if `markers_user_supplied` is not `None`
        but there are values missing from `markers_user_supplied`?

    Returns
    -------
    :
        markers with values for all values of `marker_index_level`

    Warns
    -----
    UserWarning
        `warn_on_value_missing` is `True`,
        `markers_user_supplied` is not `None`
        and there are values in `marker_index_level`
        which are not in `markers_user_supplied`.
    """
    marker_values = pandas_obj.index.get_level_values(marker_index_level).unique()

    if markers_user_supplied is None:
        # Make it all ourselves.
        # Don't warn as the user didn't set any values
        # so it is clear they want us to fill in everything.
        marker_cycler = get_default_marker_cycler()
        markers_out = {v: next(marker_cycler) for v in marker_values}

        return markers_out

    # User-supplied markers
    missing_from_user_supplied = [
        v for v in marker_values if v not in markers_user_supplied
    ]
    if not missing_from_user_supplied:
        # Just return the values we need
        return {v: markers_user_supplied[v] for v in marker_values}

    if warn_on_value_missing:
        msg = (
            f"Some marker values are not in the user-supplied markers, "
            "they will be filled from the default marker cycler instead. "
            f"{missing_from_user_supplied=} {markers_user_supplied=}"
        )
        warnings.warn(msg)

    markers_out = {}
    marker_cycler = get_default_marker_cycler()
    for v in marker_values:
        markers_out[v] = (
            markers_user_supplied[v]
            if v in markers_user_supplied
            else next(marker_cycler)
        )

    return markers_out


def fill_out_palette(
    # TODO: rename 'hue' to 'color' throughout
    # TODO: find TODOs in other 'fore-background-plot-*' branches
    pandas_obj: P,
    color_index_level: str,
    palette_user_supplied: PALETTE_LIKE[T] | None,
    warn_on_value_missing: bool,
) -> PALETTE_LIKE[T]:
    """
    Fill out a palette

    Parameters
    ----------
    pandas_obj
        Pandas object for which to fill out the palette

    color_index_level
        Index level in `pandas_obj` from which to get the values
        which require a value in the output palette

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
    color_values = pandas_obj.index.get_level_values(color_index_level).unique()

    if palette_user_supplied is None:
        # Make it all ourselves.
        # Don't warn as the user didn't set any values
        # so it is clear they want us to fill in everything.
        colour_cycler = get_default_colour_cycler()
        palette_out: PALETTE_LIKE[T] = {  # type: ignore # not sure what I've done wrong
            v: next(colour_cycler) for v in color_values
        }

        return palette_out

    # User-supplied palette
    missing_from_user_supplied = [
        v for v in color_values if v not in palette_user_supplied
    ]
    if not missing_from_user_supplied:
        # Just return the values we need
        return {v: palette_user_supplied[v] for v in color_values}  # type: ignore # not sure what mypy doesn't like

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
        for k in color_values
    }

    return palette_out


def get_default_color_var_label(color_var: str) -> str:
    """
    Get default colour variable label

    Parameters
    ----------
    color_var
        Colour variable

    Returns
    -------
    :
        Default colour variable label
    """
    color_var_label = color_var.capitalize()

    return color_var_label


def get_default_linestyle_var_label(linestyle_var: str) -> str:
    """
    Get default linestyle variable label

    Parameters
    ----------
    color_var
        linestyle variable

    Returns
    -------
    :
        Default linestyle variable label
    """
    linestyle_var_label = linestyle_var.capitalize()

    return linestyle_var_label


def get_default_marker_var_label(marker_var: str) -> str:
    """
    Get default marker variable label

    Parameters
    ----------
    marker_var
        Marker variable

    Returns
    -------
    :
        Default marker variable label
    """
    marker_var_label = marker_var.capitalize()

    return marker_var_label


def get_default_quantile_var_label(quantile_var: str) -> str:
    """
    Get default quantile variable label

    Parameters
    ----------
    quantile_var
        Quantile variable

    Returns
    -------
    :
        Default quantile variable label
    """
    quantile_var_label = quantile_var.capitalize()

    return quantile_var_label


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
