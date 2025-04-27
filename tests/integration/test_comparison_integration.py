"""
Tests of `pandas_openscm.comparison`
"""

import pandas as pd

from pandas_openscm.comparison import compare_close


def test_equal():
    # TODO: write test of what happens when we put in two equal DataFrames,
    # something like
    assert compare_close(df, df, left_name="left", right_name="right").empty


def test_equal_misaligned():
    # TODO: write test of what happens when we put in two equal DataFrames
    # that have a different row order
    assert False, "To implement"


def test_compare_close():
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
