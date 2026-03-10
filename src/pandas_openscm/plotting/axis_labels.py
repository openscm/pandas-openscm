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
    unit_aware
        Should the plot be unit-aware? Any 'truthy' value is treated as `True`
        and the y-label is not set (the unit-aware plotting will handle the y-label).

    y_label
        If `True`, we will try and infer the y-label based on the data's units
        if `unit_var` is not `None`. Otherwise, we will not try and infer.

    unit_index_level
        Index level from which unit information
        should be extracted from the plotting data.

    Returns
    -------
    :
        Whether the axis label should be inferred from the unit information or not
    """
    if isinstance(label, str) or label is None or not label:
        # Use user-supplied value
        return label

    if unit_index_level is None:
        # Nothing to generate from
        return None

    if not isinstance(label, bool):
        msg = f"Type of {label=} is not supported"
        raise TypeError(msg)

    # y_label is `True` from here on
    if unit_aware:
        # Let unit-aware plotting do its thing
        return None

    # Try to infer y-label
    if unit_index_level not in pandas_obj.index.names:
        warnings.warn(
            "Not auto-generating the _label "
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
                f"data units {units_s}",
                stacklevel=3,
            )

        label = None

    return label
