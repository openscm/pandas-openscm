"""
Tests of `pandas_openscm.indexing`, `...openscm.mi_loc`
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
from pandas_openscm.testing import check_result, convert_to_desired_type, create_test_df

try:
    import pandas_indexing as pix
except ImportError:
    pix = None

pobj_type = pytest.mark.parametrize(
    "pobj_type",
    ("DataFrame", "Series"),
)
"""
Parameterisation to use to check handling of both DataFrame and Series
"""


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


@pobj_type
def test_multi_index_lookup(pobj_type):
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
    start = convert_to_desired_type(start, pobj_type)

    locator = pd.MultiIndex.from_tuples(
        (("sb", 2), ("sa", 3), ("sa", 4)),
        names=["scenario", "id"],
    )

    if isinstance(start, pd.DataFrame):
        exp = start.iloc[[1, 2], :]
    else:
        exp = start.iloc[[1, 2]]

    res = multi_index_lookup(start, locator)

    check_result(res, exp)


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


@pobj_type
def test_index_name_aware_lookup(pobj_type):
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
    start = convert_to_desired_type(start, pobj_type)

    locator = pd.Index((2, 4), name="id")

    if isinstance(start, pd.DataFrame):
        exp = start.iloc[[1, 3], :]
    else:
        exp = start.iloc[[1, 3]]

    res = index_name_aware_lookup(start, locator)

    check_result(res, exp)


@pobj_type
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
def test_mi_loc_same_as_pandas(locator, setup_pandas_accessors, pobj_type):
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
    start = convert_to_desired_type(start, pobj_type)

    check_result(
        start.loc[locator],
        start.openscm.mi_loc(locator),
    )
