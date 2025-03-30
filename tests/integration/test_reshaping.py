"""
Integration tests of `pandas_openscm.reshaping`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.testing import create_test_df


@pytest.mark.parametrize(
    "time_col_name, time_col_name_exp",
    (
        (None, "time"),
        ("time", "time"),
        ("Time", "Time"),
        ("year", "year"),
    ),
)
def test_to_long_data_basic(setup_pandas_accessor, time_col_name, time_col_name_exp):
    kwargs = {}
    if time_col_name is not None:
        kwargs["time_col_name"] = time_col_name_exp

    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    res = start.openscm.to_long_data(**kwargs)

    exp = start.melt(ignore_index=False, var_name=time_col_name_exp).reset_index()

    pd.testing.assert_frame_equal(res, exp, check_like=True)


def test_to_long_data_nan_handling(setup_pandas_accessor):
    df = pd.DataFrame(
        [[1, np.nan, 1.2], [2.1, 10.2, np.nan]],
        columns=[2010.0, 2015.0, 2025.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "v1", "K"),
                ("sb", "v1", "K"),
            ],
            names=["scenario", "variable", "unit"],
        ),
    )

    res = df.openscm.to_long_data()
    # nans are kept
    assert res["value"].isnull().any()

    exp = df.melt(ignore_index=False, var_name="time").reset_index()

    pd.testing.assert_frame_equal(res, exp)


def test_to_long_data_nan_handling_index(setup_pandas_accessor):
    df = pd.DataFrame(
        [[1.1, 0.8, 1.2], [2.1, 10.2, 8.4]],
        columns=[2010.0, 2015.0, 2025.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", np.nan, "K"),
                ("sb", "v1", None),
            ],
            names=["scenario", "variable", "unit"],
        ),
    )

    res = df.openscm.to_long_data()
    # nans are kept
    assert res["variable"].isnull().any()
    assert res["unit"].isnull().any()

    exp = df.melt(ignore_index=False, var_name="time").reset_index()

    pd.testing.assert_frame_equal(res, exp)
