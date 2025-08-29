"""
Accessor for [pd.Index][pandas.Index] (and sub-classes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

import pandas as pd

from pandas_openscm.index_manipulation import (
    ensure_is_multiindex,
    update_levels,
    update_levels_from_other,
)

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

    def update_levels(
        self,
        updates: dict[Any, Callable[[Any], Any]],
        remove_unused_levels: bool = True,
    ) -> pd.MultiIndex:
        """
        Update the levels

        Parameters
        ----------
        updates
            Updates to apply

            Each key is the level to which the updates will be applied.
            Each value is a function which updates the level to its new values.

        remove_unused_levels
            Remove unused levels before applying the update

            Specifically, call
            [pd.MultiIndex.remove_unused_levels][pandas.MultiIndex.remove_unused_levels].

            This avoids trying to update levels that aren't being used.

        Returns
        -------
        :
            `index` with updates applied
        """
        if not isinstance(self._index, pd.MultiIndex):
            msg = (
                "This method is only intended to be used "
                "when index is an instance of `MultiIndex`. "
                f"Received {type(self._index)}"
            )
            raise TypeError(msg)

        return update_levels(
            self._index,
            updates=updates,
            remove_unused_levels=remove_unused_levels,
        )

    def update_levels_from_other(
        self,
        update_sources: dict[
            Any,
            tuple[
                Any,
                Callable[[Any], Any] | dict[Any, Any] | pd.Series[Any],
            ]
            | tuple[
                tuple[Any, ...],
                Callable[[tuple[Any, ...]], Any]
                | dict[tuple[Any, ...], Any]
                | pd.Series[Any],
            ],
        ],
        remove_unused_levels: bool = True,
    ) -> pd.MultiIndex:
        """
        Update levels based on other levels

        If the level to be updated doesn't exist, it is created.

        Parameters
        ----------
        update_sources
            Updates to apply and their source levels

            Each key is the level to which the updates will be applied
            (or the level that will be created if it doesn't already exist).

            There are two options for the values.

            The first is used when only one level is used to update the 'target level'.
            In this case, each value is a tuple of which the first element
            is the level to use to generate the values (the 'source level')
            and the second is mapper of the form used by
            [pd.Index.map][pandas.Index.map]
            which will be applied to the source level
            to update/create the level of interest.

            Each value is a tuple of which the first element
            is the level or levels (if a tuple)
            to use to generate the values (the 'source level')
            and the second is mapper of the form used by
            [pd.Index.map][pandas.Index.map]
            which will be applied to the source level
            to update/create the level of interest.

        remove_unused_levels
            Call `ini.remove_unused_levels` before updating the levels

            This avoids trying to update based on levels that aren't being used.

        Returns
        -------
        :
            `index` with updates applied
        """
        if not isinstance(self._index, pd.MultiIndex):
            msg = (
                "This method is only intended to be used "
                "when index is an instance of `MultiIndex`. "
                f"Received {type(self._index)}"
            )
            raise TypeError(msg)

        return update_levels_from_other(
            self._index,
            update_sources=update_sources,
            remove_unused_levels=remove_unused_levels,
        )
