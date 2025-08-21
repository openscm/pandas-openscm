"""
Accessor for [pd.Index][pandas.Index] (and sub-classes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import pandas as pd

from pandas_openscm.index_manipulation import ensure_is_multiindex

if TYPE_CHECKING:
    # Hmm this is somehow not correct.
    # Figuring it out is a job for another day
    Idx = TypeVar("Idx", bound=pd.Index[Any])


else:
    Idx = TypeVar("Idx")


class PandasIndexOpenSCMAccessor(Generic[Idx]):
    """
    [pd.Index][pandas.Index] accessor

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, index: Idx):
        """
        Initialise

        Parameters
        ----------
        index
            [pd.Index][pandas.Index] to use via the accessor
        """
        # It is possible to validate here.
        # However, it's probably better to do validation closer to the data use.
        self._index = index

    def ensure_is_multiindex(self) -> pd.MultiIndex:
        """
        Ensure that the index is a [pd.MultiIndex][pandas.MultiIndex]

        Returns
        -------
        :
            `index` as a [pd.MultiIndex][pandas.MultiIndex]

            If the index was already a [pd.MultiIndex][pandas.MultiIndex],
            this is a no-op.
        """
        res = ensure_is_multiindex(self._index)

        return res

    def eim(self) -> pd.MultiIndex:
        """
        Ensure that the index is a [pd.MultiIndex][pandas.MultiIndex]

        Alias for [ensure_is_multiindex][pandas_openscm.index_manipulation.]

        Returns
        -------
        :
            `index` as a [pd.MultiIndex][pandas.MultiIndex]

            If the index was already a [pd.MultiIndex][pandas.MultiIndex],
            this is a no-op (although the value of copy is respected).
        """
        return self.ensure_is_multiindex()
