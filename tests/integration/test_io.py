"""
Integration tests of `pandas_openscm.io`
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from pandas_openscm.io import load_timeseries_csv
from pandas_openscm.testing import create_test_df


@pytest.mark.parametrize(
    "index_columns",
    (
        ["variable", "scenario", "run", "unit"],
        ["scenario", "variable", "unit", "run"],
        ["variable", "scenario", "run", "unit", "1990.0", "2005.0"],
        ["scenario", "run", "unit", "1990.0", "2005.0"],
    ),
)
def test_load_timeseries_csv_basic(tmp_path, index_columns):
    out_path = tmp_path / "test_load_timeseries_csv.csv"

    timepoints = np.arange(1990.0, 2010.0 + 1.0)
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=timepoints,
    )

    # Write in a way that definitely doesn't preserve index information
    # or column type.
    start.reset_index().to_csv(out_path, index=False)

    loaded = load_timeseries_csv(out_path, index_columns=index_columns)

    assert loaded.index.names == index_columns
    # No mangling done
    assert all(isinstance(c, str) for c in loaded.columns.values)

    all_cols_as_str = [str(v) for v in [*start.index.names, *timepoints]]
    non_index_cols = set(all_cols_as_str) - set(index_columns)
    loaded_cols = loaded.columns.tolist()
    assert all(str(v) in loaded_cols for v in non_index_cols)


@pytest.mark.parametrize("lower_column_names", (True, False))
def test_load_timeseries_csv_lower_column_names(tmp_path, lower_column_names):
    out_path = tmp_path / "test_load_timeseries_csv_lower_column_names.csv"

    timepoints = np.arange(1990.0, 2010.0 + 1.0)
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=timepoints,
    )

    # Write with capitalised columns
    to_write = start.reset_index()
    to_write.columns = [
        v.capitalize() if isinstance(v, str) else v for v in to_write.columns
    ]
    to_write.to_csv(out_path, index=False)

    if lower_column_names:
        index_columns = ["variable", "scenario", "run", "unit"]
    else:
        index_columns = ["Variable", "Scenario", "Run", "Unit"]

    loaded = load_timeseries_csv(
        out_path, index_columns=index_columns, lower_column_names=lower_column_names
    )

    assert loaded.index.names == index_columns
    assert all(isinstance(c, str) for c in loaded.columns.values)
    assert all(str(v) in loaded.columns for v in timepoints)


@pytest.mark.parametrize(
    # Column type and value type are not the same
    # because columns are held as numpy arrays.
    "out_columns_type, exp_column_value_type",
    (
        (int, np.int64),
        (float, np.float64),
        (np.float64, np.float64),
        (np.float32, np.float32),
    ),
)
def test_load_timeseries_csv_basic_out_columns_type(
    tmp_path, out_columns_type, exp_column_value_type
):
    out_path = tmp_path / "test_load_timeseries_csv.csv"

    timepoints = np.arange(1990.0, 2010.0 + 1.0, dtype=int)
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=timepoints,
    )

    # Write in a way that definitely doesn't preserve index information
    # or column type.
    start.reset_index().to_csv(out_path, index=False)

    index_columns = ["variable", "scenario", "run", "unit"]

    loaded = load_timeseries_csv(
        out_path, index_columns=index_columns, out_columns_type=out_columns_type
    )

    assert loaded.index.names == index_columns
    assert all(isinstance(c, exp_column_value_type) for c in loaded.columns.values)


@pytest.mark.parametrize(
    "out_columns_name, exp_columns_name",
    (
        (None, None),
        ("hi", "hi"),
        ("time", "time"),
    ),
)
def test_load_timeseries_csv_basic_out_columns_name(
    tmp_path, out_columns_name, exp_columns_name
):
    out_path = tmp_path / "test_load_timeseries_csv.csv"

    timepoints = np.arange(1990.0, 2010.0 + 1.0, dtype=int)
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=timepoints,
    )
    assert start.columns.name is None

    start.to_csv(out_path)

    index_columns = ["variable", "scenario", "run", "unit"]

    loaded = load_timeseries_csv(
        out_path, index_columns=index_columns, out_columns_name=out_columns_name
    )

    assert loaded.columns.name == exp_columns_name


@pytest.mark.xfail(reason="Not implemented")
def test_load_timeseries_csv_infer_index_cols(tmp_path):
    # Suggested cases here:
    # - datetime columns
    # - int columns
    # - float columns
    # - array columns
    out_path = tmp_path / "test_load_timeseries_csv.csv"

    timepoints = np.arange(1990.0, 2010.0 + 1.0)
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=timepoints,
    )

    # Write in a way that definitely doesn't preserve index information
    # or column type and shuffles the columns order.
    to_write = start.reset_index()
    cols = to_write.columns.tolist()
    random.shuffle(cols)
    to_write = to_write[cols]
    to_write.to_csv(out_path, index=False)

    loaded = load_timeseries_csv(out_path)

    exp_index_columns = ["scenario", "variable", "unit", "run"]
    assert loaded.index.names == exp_index_columns
    assert all(isinstance(c, str) for c in loaded.columns.values)
    assert all(str(v) in loaded.columns for v in timepoints)
