"""
Tests of `pandas_openscm.io`
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
    assert all(isinstance(c, str) for c in loaded.columns)
    assert all(str(v) in loaded.columns for v in timepoints)


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
    to_write.columns = to_write.columns.str.capitalize()
    to_write.to_csv(out_path, index=False)

    if lower_column_names:
        index_columns = ["variable", "scenario", "run", "unit"]
    else:
        index_columns = ["Variable", "Scenario", "Run", "Unit"]

    loaded = load_timeseries_csv(
        out_path, index_columns=index_columns, lower_column_names=lower_column_names
    )

    assert loaded.index.names == index_columns
    assert all(isinstance(c, str) for c in loaded.columns)
    assert all(str(v) in loaded.columns for v in timepoints)


@pytest.mark.parametrize("out_column_type", (int, float, np.float64))
def test_load_timeseries_csv_basic_out_column_type(tmp_path, out_column_type):
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

    index_columns = ["variable", "scenario", "run", "unit"]

    loaded = load_timeseries_csv(
        out_path, index_columns=index_columns, out_column_type=out_column_type
    )

    assert loaded.index.names == index_columns
    assert all(isinstance(c, out_column_type) for c in loaded.columns)


@pytest.mark.xfail(reason="Not implemented")
def test_load_timeseries_csv_infer_index_cols(tmp_path):
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
    assert all(isinstance(c, str) for c in loaded.columns)
    assert all(str(v) in loaded.columns for v in timepoints)
