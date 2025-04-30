"""
Tests of `pandas_openscm.comparison`
"""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.comparison import compare_close


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        data=[[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
        columns=pd.Index([2.0, 4.0, 10.0], name="time"),
        index=pd.MultiIndex.from_tuples(
            [("v1", "kg"), ("v2", "m"), ("v3", "yr")], names=["variable", "unit"]
        ),
    )


def test_equal(simple_df):
    assert compare_close(
        simple_df, simple_df, left_name="left", right_name="right"
    ).empty


def test_equal_misaligned(simple_df):
    left = simple_df

    # switch rows 2 and 3
    right = left.iloc[[0, 2, 1], :].copy()

    assert compare_close(left, right, left_name="left", right_name="right").empty


def create_test_df(
    data,
    columns=(2.0, 4.0, 10.0),
    index=(("v1", "kg"), ("v2", "m"), ("v3", "yr")),
):
    return pd.DataFrame(
        data=data,
        columns=pd.Index(columns, name="time"),
        index=pd.MultiIndex.from_tuples(index, names=["variable", "unit"]),
    )


@pytest.mark.parametrize(
    "left, right, left_name, right_name, isclose, exp",
    (
        pytest.param(
            create_test_df(
                data=[[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
            ),
            create_test_df(
                data=[[1.5, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
            ),
            "left",
            "right",
            partial(np.isclose, rtol=0.1),
            pd.DataFrame(
                data=[[1.0, 1.5]],
                columns=["left", "right"],
                index=pd.MultiIndex.from_tuples(
                    [("v1", "kg", 2.0)], names=["variable", "unit", "time"]
                ),
            ),
            id="values outside relative tolerance",
        ),
        pytest.param(
            create_test_df(
                data=[[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
            ),
            create_test_df(
                data=[[1.5, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
            ),
            "name1_left",
            "name2_right",
            partial(np.isclose, rtol=0.6),
            pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=["variable", "unit", "time"]),
                columns=["name1_left", "name2_right"],
            ),
            id="values within relative tolerance",
        ),
        pytest.param(
            create_test_df(
                data=[[1.0, 2.0, 3.0], [1.1, 1.2, 1.7], [-1.1, 0.0, 0.5]],
            ),
            create_test_df(
                data=[[1.5, 2.0, 3.5], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
            ),
            "name1_left",
            "name2_right",
            partial(np.isclose, atol=0.5),
            pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=["variable", "unit", "time"]),
                columns=["name1_left", "name2_right"],
            ),
            id="values within absolute tolerance - edge case",
        ),
    ),
)
def test_compare_close(left, right, left_name, right_name, isclose, exp):  # noqa: PLR0913
    pd.testing.assert_frame_equal(
        compare_close(left, right, left_name, right_name, isclose),
        exp,
        #  when comparing empty dataframes, index type and dtype may be different
        check_index_type=False,
        check_dtype=False,
    )
