"""
Tests of `pandas_openscm.indexing` and `pd.DataFrame.openscm.mi_loc`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.indexing import (
    index_name_aware_lookup,
    index_name_aware_match,
    multi_index_lookup,
    multi_index_match,
)
from pandas_openscm.testing import create_test_df

try:
    import pandas_indexing as pix
except ImportError:
    pix = None


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
    "start, locator, exp",
    (
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sc", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.Index(["ma", "mb"], name="model"),
            [True, True, True, True],
            id="first-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sc", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.Index(["sa", "sb"], name="scenario"),
            [True, True, False, True],
            id="second-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                (
                    ("ma", "sa", 1),
                    ("ma", "sb", 2),
                    ("mb", "sc", 3),
                    ("mb", "sb", 4),
                ),
                names=["model", "scenario", "id"],
            ),
            pd.Index(["sa", "sb"], name="scenario"),
            [True, True, False, True],
            id="third-level",
        ),
    ),
)
def test_index_name_aware_match(start, locator, exp):
    res = index_name_aware_match(start, locator)
    # # # If you want to see what fails with plain pandas, use the below
    # res = start.isin(locator)
    np.testing.assert_equal(res, exp)


def test_index_name_aware_lookup():
    # Most of the tests are in test_index_name_aware_match.
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

    locator = pd.Index((2, 4), name="id")

    exp = start.iloc[[1, 3], :]

    res = index_name_aware_lookup(start, locator)

    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.parametrize(
    "locator",
    (
        pytest.param(["scenario_2", "scenario_1"], id="list"),
        pytest.param(pd.Index(["scenario_2", "scenario_1"]), id="index-no-name"),
        pytest.param(
            ["variable_2", "variable_3"],
            id="list-second-level",
            marks=pytest.mark.xfail(
                reason="pandas looks up the first level rather than variables"
            ),
        ),
        pytest.param(
            pd.Index(["variable_2", "variable_3"]),
            id="index-no-name-second-level",
            marks=pytest.mark.xfail(
                reason="pandas looks up the first level rather than variables"
            ),
        ),
        pytest.param(
            pix.isin(scenario=["scenario_1", "scenario_3"])
            if pix is not None
            else None,
            id="pix_isin",
            marks=pytest.mark.skipif(pix is None, reason="pandas-indexing unavailable"),
        ),
        pytest.param(
            pix.ismatch(
                scenario=[
                    "*1",
                ]
            )
            if pix is not None
            else None,
            id="pix_ismatch",
            marks=pytest.mark.skipif(pix is None, reason="pandas-indexing unavailable"),
        ),
    ),
)
def test_mi_loc_same_as_pandas(locator, setup_pandas_accessor):
    """
    Test pass through in the cases where pass through should happen

    For the cases where there shouldn't be pass through,
    see `test_multi_index_match`
    and `test_index_name_aware_match`.
    """
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
