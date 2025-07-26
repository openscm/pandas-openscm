"""
API for [pandas][] accessors.

**Notes for developers**

We generally don't like using mix-ins.
However, the pandas accessors requires using a class,
so mix-ins are kind of the only way to do this
without significant duplication
(although, if you see a different way,
particularly something more composition based, PRs welcome!).

The mix-ins are defined in various modules,
then combined to make the accessors
for [DataFrame][pandas.DataFrame]'s, [Series][pandas.Series]'s
and [Index][pandas.Index]'s.

We try and keep the accessors as a super-thin layer.
This makes it easier to re-use functionality in a functional way,
which is beneficial
(particularly if we one day need to switch to
a different kind of dataframe e.g. dask).

As a result, we effectively duplicate our API in the mix-ins.
This is fine, because this repo is not so big.
Pandas and pandas-indexing use pandas' `pandas.util._decorators.docs` decorator
(see https://github.com/pandas-dev/pandas/blob/05de25381f71657bd425d2c4045d81a46b2d3740/pandas/util/_decorators.py#L342)
to avoid duplicating the docs.
We could use the same pattern, but I have found that this magic
almost always goes wrong so I would stay away from this as long as we can.
"""

from __future__ import annotations

import pandas as pd

from pandas_openscm.accessors.dataframe import PandasDataFrameOpenSCMAccessor


# TODO: note change in name (now has trailing s) in changelog
def register_pandas_accessors(namespace: str = "openscm") -> None:
    """
    Register the pandas accessors

    This registers accessors
    for [DataFrame][pandas.DataFrame]'s, [Series][pandas.Series]'s
    and [Index][pandas.Index]'s.
    If you only want to register accessors for one of these,
    we leave it up to you to copy the line(s) you need.

    For details of how these accessors work, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

    We provide this as a separate function
    because we have had really bad experiences with imports having side effects
    (which seems to be the more normal pattern)
    and don't want to pass those bad experiences on.

    Parameters
    ----------
    namespace
        Namespace to use for the accessor

        E.g. if namespace is 'custom'
        then the pandas-openscm API will be available under
        `pd.DataFrame.custom.pandas_openscm_function`
        e.g. `pd.DataFrame.custom.convert_unit`.
    """
    pd.api.extensions.register_dataframe_accessor(namespace)(
        PandasDataFrameOpenSCMAccessor
    )
    # pd.api.extensions.register_series_accessor(namespace)(PandasSeriesOpenSCMAccessor)
    # pd.api.extensions.register_index_accessor(namespace)(PandasIndexOpenSCMAccessor)
