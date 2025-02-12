"""
API for [`pandas`][pandas] accessors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

if TYPE_CHECKING:
    P = TypeVar("P", bound=pd.DataFrame | pd.Series[Any])


class DataFramePandasOpenSCMAccessor:
    """
    [`pd.DataFrame`][pandas.DataFrame] accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialise

        Parameters
        ----------
        pandas_obj
            Pandas object to use via the accessor
        """
        # It is possible to validate here.
        # However, it's probably better to do validation closer to the data use.
        self._df = pandas_obj

    def to_category_index(
        self,
    ) -> pd.DataFrame:
        """
        Convert the index's values to categories

        This can save a lot of memory and improve the speed of processing.
        However, it comes with some pitfalls.
        For a nice discussion of some of them,
        see [this article](https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a/).

        Returns
        -------
        :
            [`pd.DataFrame`][pandas.DataFrame] with all index columns
            converted to category type.
        """
        new_index = pd.MultiIndex.from_frame(
            self._df.index.to_frame(index=False).astype("category")
        )

        return pd.DataFrame(
            self._df.values,
            index=new_index,
            columns=self._df.columns,
        )


def register_pandas_accessor(namespace: str = "openscm") -> None:
    """
    Register the pandas accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

    We provide this as a separate function
    because we have had really bad experiences with imports having side effects
    and don't want to pass those on to our users.

    Parameters
    ----------
    namespace
        Namespace to use for the accessor
    """
    pd.api.extensions.register_dataframe_accessor(namespace)(
        DataFramePandasOpenSCMAccessor
    )
