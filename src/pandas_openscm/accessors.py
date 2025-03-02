"""
API for [pandas][] accessors.

As a general note to developers,
we try and keep the accessors as a super-thin layer.
This makes it easier to re-use functionality in a more functional way,
which is beneficial
(particularly if we one day need to switch to
a different kind of dataframe e.g. dask).

As a result, we effectively duplicate our API.
This is fine, because this repo is not so big.
Pandas and pandas-indexing use pandas' `pandas.util._decorators.docs` decorator
(see https://github.com/pandas-dev/pandas/blob/05de25381f71657bd425d2c4045d81a46b2d3740/pandas/util/_decorators.py#L342)
to avoid duplicating the docs.
We could use the same pattern, but I have found that this magic
almost always goes wrong so I would stay away from this as long as we can.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from pandas_openscm.index_manipulation import convert_index_to_category_index
from pandas_openscm.indexing import mi_loc

if TYPE_CHECKING:
    import pandas_indexing as pix


class DataFramePandasOpenSCMAccessor:
    """
    [pd.DataFrame][pandas.DataFrame] accessors

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

    def mi_loc(
        self,
        locator: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector,
    ) -> pd.DataFrame:
        """
        Select data, being slightly smarter than the default [pandas.DataFrame.loc][].

        Parameters
        ----------
        locator
            Locator to apply

            If this is a multi-index, we use
            [multi_index_lookup][(p).indexing.] to ensure correct alignment.

            If this is an index that has a name,
            we use the name to ensure correct alignment.

        Returns
        -------
        :
            Selected data

        Notes
        -----
        If you have [pandas_indexing][] installed,
        you can get the same (perhaps even better) functionality
        using something like the following instead

        ```python
        ...
        pandas_obj.loc[pandas_indexing.isin(locator)]
        ...
        ```
        """
        return mi_loc(self._df, locator)

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
            [pd.DataFrame][pandas.DataFrame] with all index columns
            converted to category type.
        """
        return convert_index_to_category_index(self._df)


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
