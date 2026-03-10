"""
Axis label logic and helpers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint


def infer_axis_label_from_unit_information(
    unit_aware: bool | pint.facets.PlainRegistry,
    y_label: str | bool | None,
    unit_index_level: str | None,
) -> bool:
    """
    Infer axis label

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
    if unit_aware:
        return False

    infer_label = isinstance(y_label, bool) and y_label and unit_index_level is not None

    return infer_label
