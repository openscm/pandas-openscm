"""
Axis label logic and helpers
"""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

if TYPE_CHECKING:
    import pandas as pd
    import pint

    P = TypeVar("P", pd.DataFrame, pd.Series[Any])


def handle_axis_label_inference_from_unit_information(
    label: str | bool | None,
    unit_aware: bool | pint.facets.PlainRegistry,
    pandas_obj: P,
    unit_index_level: str | None,
    warn_infer_label_with_multi_unit: bool,
) -> str | None:
    """
    Handle logic related to inferring axis label from unit information

    Parameters
    ----------
    label
        Label to use.

        If a string or `None`, this is returned as-is.

        If `True` and `not unit_aware`, then we will attempt to generate the label.

        If `False`, then `None` is returned.

    unit_aware
        Is the plot being done unit-aware?

        If `y_label` is `True`, then we will return `None`
        so that the unit-aware plotting can control the y-label value.

    pandas_obj
        Pandas object from which to get unit values if needed.

    unit_index_level
        Index level from which unit information should be extracted if needed.

    warn_infer_label_with_multi_unit
            Should a warning be raised if we try to infer the unit
            but the data has more than one unit?

    Returns
    -------
    :
        Axis label to use when plotting.
    """
    if not infer_label(label):
        return cast_label_false_to_none(label)

    # label should be `True` from here on
    if not (isinstance(label, bool) and label):  # pragma: no cover
        msg = f"`label` should be `True` at this point. Have {label=}"
        raise AssertionError(msg)

    return try_to_get_unit_label(
        unit_aware=unit_aware,
        pandas_obj=pandas_obj,
        unit_index_level=unit_index_level,
        warn_infer_label_with_multi_unit=warn_infer_label_with_multi_unit,
    )


def infer_label(label: str | bool | None) -> bool:
    """
    Return whether a label should be inferred or not

    Parameters
    ----------
    label
        Label input value

        If a string or `None` or `False`,
        a label should not be inferred (i.e. return `False`).

        If `True`,
        a label should be inferred (i.e. return `True`).

    Returns
    -------
    :
        Whether a label should be inferred or not.
    """
    if isinstance(label, bool) and label:
        return True

    return False


def cast_label_false_to_none(label: str | bool | None) -> str | None:
    """
    Cast a value of `False` for `label` to `None`.

    Saves us writing this casting all over the place.

    Parameters
    ----------
    label
        Label input value

    Returns
    -------
    :
        Cast value of `label`
        (`label` is returned unchanged unless `label` is `False`,
        in which case `None` is returned).
    """
    if isinstance(label, bool) and not label:
        return None

    if isinstance(label, bool) and label:  # pragma: no cover
        msg = (
            "If `label` is a boolean, it should be `False` when calling this function. "
            "Please check whether you need to do this casting "
            "by calling `infer_label` first. "
            f"Received {label=}"
        )
        raise AssertionError(msg)

    return label


def try_to_get_unit_label(
    unit_aware: bool | pint.facets.PlainRegistry,
    pandas_obj: P,
    unit_index_level: str | None,
    warn_infer_label_with_multi_unit: bool,
) -> str | None:
    """
    Try to get unit label from data

    Parameters
    ----------
    unit_aware
        Is the plot being done unit-aware?

        If this value is truthy, then we will return `None`
        so that the unit-aware plotting can control the unit label value.

    pandas_obj
        Pandas object from which to get unit values if needed.

    unit_index_level
        Index level from which unit information should be extracted if needed.

    warn_infer_label_with_multi_unit
            Should a warning be raised if we try to infer the unit
            but the data has more than one unit?

    Returns
    -------
    :
        Unit label to use when plotting
        or `None` if an unambiguous unit label couldn't be inferred
        or we are using unit-aware plotting.
    """
    if unit_index_level is None:
        # Nothing to generate from
        return None

    if unit_aware:
        # Let unit-aware plotting do its thing,
        # don't need to infer ourselves.
        return None

    if unit_index_level not in pandas_obj.index.names:
        warnings.warn(
            "Not auto-generating the label "
            f"because {unit_index_level=} is not in {pandas_obj.index.names=}",
            stacklevel=3,
        )
        return None

    values_units = pandas_obj.index.get_level_values(unit_index_level)
    units_s = set(values_units)
    if len(units_s) == 1:
        label = values_units[0]
    else:
        # More than one unit plotted, don't infer a label
        if warn_infer_label_with_multi_unit:
            warnings.warn(
                "Not auto-generating the label "
                "because the data has more than one unit: "
                f"data units {sorted(units_s)}",
                stacklevel=3,
            )

        label = None

    return label
