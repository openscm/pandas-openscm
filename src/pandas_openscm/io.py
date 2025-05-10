"""
Serialisation/deserialisation (i.e. input/output) support
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_timeseries_csv(
    fp: Path,
    lower_column_names: bool = True,
    index_columns: list[str] | None = None,
    out_columns_type: type | None = None,
    out_columns_name: str | None = None,
) -> pd.DataFrame:
    """
    Load a CSV holding timeseries

    In other words, a CSV that has metadata columns
    and then some time columns.

    Parameters
    ----------
    fp
        File path to load

    lower_column_names
        Convert the column names to all lower case as part of loading.

        Note, if `lower_col_names` is `True`,
        the column names are converted to lower case
        before the index is set so
        a) you should only use lower case in `index_columns`
        and b) the lowering will affect values that do not end up in the index too.

    index_columns
        Columns to treat as metadata from the loaded CSV.

        At the moment, if this is not provided, a `NotImplementedError` is raised.
        In future, if not provided, we will try and infer the columns
        based on whether they look like time columns or not.

    out_columns_type
        The type to apply to the output columns that are not part of the index.

        If not supplied, the raw type returned by pandas is returned.

    out_columns_name
        The name for the columns in the output.

        If not supplied, the raw name returned by pandas is returned.

        This can also be set with
        [pd.DataFrame.rename_axis][pandas.DataFrame.rename_axis]
        but we provide it here for convenience
        (and in case you couldn't find this trick for ages, like us).

    Returns
    -------
    :
        Loaded data
    """
    out = pd.read_csv(fp)

    if lower_column_names:
        out.columns = out.columns.str.lower()

    if index_columns is None:
        raise NotImplementedError(index_columns)

    out = out.set_index(index_columns)

    if out_columns_type is not None:
        out.columns = out.columns.astype(out_columns_type)

    if out_columns_name is not None:
        out = out.rename_axis(out_columns_name, axis="columns")

    return out
