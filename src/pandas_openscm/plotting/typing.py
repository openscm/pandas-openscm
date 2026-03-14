"""
Types used across this module
"""

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    T = TypeVar("T")

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

    DASH_VALUE_LIKE: TypeAlias = Union[str, tuple[float, tuple[float, ...]]]
    """Types that allow a dash to be specified in matplotlib"""

    class PALETTE_LIKE(
        Generic[T],
        Mapping[T, COLOUR_VALUE_LIKE],
    ):
        """Palette-like type"""

    QUANTILES_PLUMES_LIKE: TypeAlias = tuple[
        Union[tuple[float, float], tuple[tuple[float, float], float]], ...
    ]
    """Type that quantiles and the alpha to use for plotting their line/plume"""

    class HasXVals(Protocol):
        """Object that has x-values"""

        x_vals: NP_ARRAY_OF_FLOAT_OR_INT | PINT_NUMPY_ARRAY
        """x-values to plot"""
