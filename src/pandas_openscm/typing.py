"""
Type hints that are used throughout
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pint
    from typing_extensions import TypeAlias

NP_FLOAT_OR_INT: TypeAlias = Union[np.floating[Any], np.integer[Any]]
"""
Type alias for a numpy float or int (not complex)
"""

NP_ARRAY_OF_FLOAT_OR_INT: TypeAlias = npt.NDArray[NP_FLOAT_OR_INT]
"""
Type alias for an array of numpy float or int (not complex)
"""

NP_ARRAY_OF_BOOL: TypeAlias = npt.NDArray[np.bool]
"""
Type alias for an array of boolean values
"""

if TYPE_CHECKING:
    PINT_NUMPY_ARRAY: TypeAlias = pint.facets.numpy.quantity.NumpyQuantity[
        NP_ARRAY_OF_FLOAT_OR_INT
    ]
    """
    Type alias for a pint quantity that wraps a numpy array

    No shape hints because that doesn't seem to be supported by numpy yet.
    """

    PINT_SCALAR: TypeAlias = pint.facets.numpy.quantity.NumpyQuantity[NP_FLOAT_OR_INT]
    """
    Type alias for a pint quantity that wraps a numpy scalar
    """
