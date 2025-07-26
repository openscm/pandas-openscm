"""
The protocol that is assumed by mix-in's.

This allows us to ensure that the contract between our accessors
and the mix-in's we used is met.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    import pandas as pd

    P = TypeVar(
        "P",
        pd.DataFrame,
        pd.Series[str],
        pd.Series[float],
        pd.Series[int],
        pd.Series[Any],
        pd.Index[Any],
    )


class PandasAccessorLike(Protocol[P]):
    """
    Class that has the properties of a pandas accessor
    """

    pandas_obj: P
    """Pandas object being used via the accessor"""
