"""
Tests of `pandas_openscm.comparison`
"""

import pandas as pd
import pytest

from pandas_openscm.comparison import compare_close
from functools import partial
import numpy as np

@pytest.fixture
def test_df():
    return pd.DataFrame(
        data=[[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
        columns = pd.Index([2.0, 4.0, 10.0], name="time"),
        index = pd.MultiIndex.from_tuples(
 [("v1", "kg"), ("v2", "m"), ("v3", "yr")], names=["variable", "unit"]
        ),
    )

def test_equal(test_df):

    assert compare_close(test_df, test_df, left_name="left", right_name="right").empty


def test_equal_misaligned(test_df):

    left = test_df

    # randomly shuffle the all rows (with random seed so we can reproduce)
    right = test_df.sample(frac=1, random_state=1)

    assert compare_close(left, right, left_name="left", right_name="right").empty

@pytest.mark.parametrize(
    "left, right, left_name, right_name, isclose, exp",
    (
        (
        pd.DataFrame(
                    data=[[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
                    columns=pd.Index([2.0, 4.0, 10.0], name="time"),
                    index=pd.MultiIndex.from_tuples(
                        [("v1", "kg"), ("v2", "m"), ("v3", "yr")], names=["variable", "unit"]
                    ),
                ),
        pd.DataFrame(
            data=[[1.5, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
            columns=pd.Index([2.0, 4.0, 10.0], name="time"),
            index=pd.MultiIndex.from_tuples(
                [("v1", "kg"), ("v2", "m"), ("v3", "yr")], names=["variable", "unit"]
            ),
        ),
            "left",
            "right",
            partial(np.isclose, rtol=0.1),
        pd.DataFrame(
            data=[[1.5, 2.0]],
            columns=["left", "right"],
            index=pd.MultiIndex.from_tuples(
                [("v1", "kg", 2.0), ], names=["variable", "unit", "time"]
            ),
        ),
        ),
    ),
)
def test_compare_close(left, right, left_name, right_name, isclose, exp):
    # TODO: use parametrize to test a series of cases.
    # In each case, we should specify left, right, left_name, right_name,
    # isclose and our expectation (exp).
    # Test for cases where we expect a difference and don't expect a difference,
    # also using different tolerances (e.g. for a given input,
    # make sure that the tolerance determines whether we get differences or not).
    # Also check that the given left_name and right_name are respected.

    # Then the test should simply look like
    pd.testing.assert_frame_equal(
        compare_close(left, right, left_name, right_name, isclose), exp
    )

# def test_compare_close():
#     # TODO: use parametrize to test a series of cases.
#     # In each case, we should specify left, right, left_name, right_name,
#     # isclose and our expectation (exp).
#     # Test for cases where we expect a difference and don't expect a difference,
#     # also using different tolerances (e.g. for a given input,
#     # make sure that the tolerance determines whether we get differences or not).
#     # Also check that the given left_name and right_name are respected.
#
#     # Then the test should simply look like
#     pd.testing.assert_frame_equal(
#         compare_close(left, right, left_name, right_name, isclose), exp
#     )
