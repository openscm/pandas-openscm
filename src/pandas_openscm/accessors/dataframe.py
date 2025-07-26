"""
[DataFrame][pandas.DataFrame] accessor
"""

from __future__ import annotations

import pandas as pd

from pandas_openscm.accessors.unit_conversion import UnitConversionMixIn


class PandasDataFrameOpenSCMAccessor(
    UnitConversionMixIn[pd.DataFrame],
):
    """
    [pd.DataFrame][pandas.DataFrame] accessor

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialise

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame] to use via the accessor
        """
        # It is possible to validate here.
        # However, it's probably better to do validation closer to the data use
        # given how varied our use cases can be.
        self.pandas_obj = df
