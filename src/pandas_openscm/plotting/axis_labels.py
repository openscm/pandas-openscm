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
    if isinstance(label, str) or label is None or not label:
        # Use user-supplied value
        return label

    if unit_index_level is None:
        # Nothing to generate from
        return None

    if not isinstance(label, bool):
        msg = f"{type(label)} are not supported. {label=}"
        raise TypeError(msg)

    # y_label is `True` from here on
    if unit_aware:
        # Let unit-aware plotting do its thing
        return None

    # Try to infer y-label
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
        # More than one unit plotted, don't infer a y-label
        if warn_infer_label_with_multi_unit:
            warnings.warn(
                "Not auto-generating the label "
                "because the data has more than one unit: "
                f"data units {sorted(units_s)}",
                stacklevel=3,
            )

        label = None

    return label
