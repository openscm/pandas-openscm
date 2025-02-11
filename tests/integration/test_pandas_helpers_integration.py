"""
Tests of `pandas_openscm.pandas_helpers`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.pandas_helpers import multi_index_lookup, multi_index_match


@pytest.mark.parametrize(
    "start, locator, exp",
    (
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sa", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("mb", "sa", 3),
                ),
                names=["model", "scenario", "id"],
            ),
            [True, False, True, False],
            id="all-levels-covered",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sa", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.MultiIndex.from_tuples(
                (("ma",),),
                names=["model"],
            ),
            [True, True, False, False],
            id="only-first-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sa", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.MultiIndex.from_tuples(
                (("sa",),),
                names=["scenario"],
            ),
            [True, False, True, False],
            id="only-second-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sa", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.MultiIndex.from_tuples(
                (("ma", 1), ("mb", 4)),
                names=["model", "id"],
            ),
            [True, False, False, True],
            id="first-and-third-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sa", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.MultiIndex.from_tuples(
                (("sb", 2), ("sa", 3)),
                names=["scenario", "id"],
            ),
            [False, True, True, False],
            id="second-and-third-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sa", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.MultiIndex.from_tuples(
                (("sb", 2), ("sa", 4)),
                names=["scenario", "id"],
            ),
            [False, True, False, False],
            id="second-and-third-level-not-all-present",
        ),
    ),
)
def test_multi_index_match(start, locator, exp):
    res = multi_index_match(start, locator)
    # # If you want to see what fails with plain pandas, use the below
    # res = start.isin(locator)
    np.testing.assert_equal(res, exp)


def test_multi_index_lookup():
    # Most of the tests are in test_multi_index_match.
    # Hence why there is only one here.
    start = pd.DataFrame(
        np.arange(8).reshape((4, 2)),
        columns=[2010, 2020],
        index=pd.MultiIndex.from_tuples(
            (
                ("ma", "sa", 1),
                ("ma", "sb", 2),
                ("mb", "sa", 3),
                ("mb", "sb", 4),
            ),
            names=["model", "scenario", "id"],
        ),
    )

    locator = pd.MultiIndex.from_tuples(
        (("sb", 2), ("sa", 3), ("sa", 4)),
        names=["scenario", "id"],
    )

    exp = start.iloc[[1, 2], :]

    res = multi_index_lookup(start, locator)

    pd.testing.assert_frame_equal(res, exp)
