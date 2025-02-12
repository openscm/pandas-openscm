"""
Tests of `pandas_openscm.indexing` and `pd.DataFrame.openscm.mi_loc`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pytest

from pandas_openscm.accessors import register_pandas_accessor
from pandas_openscm.indexing import multi_index_lookup, multi_index_match
from pandas_openscm.testing import create_test_df


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


@pytest.mark.parametrize(
    "locator",
    (
        pytest.param(pix.isin(scenario=["scenario_1", "scenario_3"]), id="pix_isin"),
        pytest.param(
            pix.ismatch(
                scenario=[
                    "*1",
                ]
            ),
            id="pix_ismatch",
        ),
    ),
)
def test_mi_loc_same_as_pandas(locator):
    register_pandas_accessor()

    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.arange(1990.0, 2010.0 + 1.0),
    )

    pd.testing.assert_frame_equal(
        start.loc[locator],
        start.openscm.mi_loc(locator),
    )
