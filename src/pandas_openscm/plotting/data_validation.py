"""
Data validation helpers
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    import attr

    from pandas_openscm.plotting.typing import (
        HasXVals,
    )
    from pandas_openscm.typing import NP_ARRAY_OF_FLOAT_OR_INT, PINT_NUMPY_ARRAY


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
