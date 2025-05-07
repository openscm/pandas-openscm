"""
Test `pandas_openscm.index_manipulation.set_levels`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.index_manipulation import set_index_levels, set_levels


@pytest.mark.parametrize(
    "start, levels_to_set, exp",
    (
        pytest.param(
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 0),
                    ("sb", "vb", "m", 1),
                    ("sa", "va", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            {"new_variable": "test"},
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 0, "test"),
                    ("sb", "vb", "m", 1, "test"),
                    ("sa", "va", "kg", 2, "test"),
                ],
                names=["scenario", "variable", "unit", "run_id", "new_variable"],
            ),
            id="set-single-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 0),
                    ("sb", "vb", "m", 1),
                    ("sa", "va", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            {"new_variable": ["a", "b", "c"]},
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 0, "a"),
                    ("sb", "vb", "m", 1, "b"),
                    ("sa", "va", "kg", 2, "c"),
                ],
                names=["scenario", "variable", "unit", "run_id", "new_variable"],
            ),
            id="set-multiple-values",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 5),
                    ("sb", "vb", "m", 1),
                    ("sa", "va", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            {"variable": ["a", "b", "c"]},
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "a", "kg", 5),
                    ("sb", "b", "m", 1),
                    ("sa", "c", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            id="replace-existing-level",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 5),
                    ("sb", "vb", "m", 1),
                    ("sa", "va", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            {"new_variable": ["a", "b", "c"], "another_new_variable": [0, 0, 0]},
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 5, "a", 0),
                    ("sb", "vb", "m", 1, "b", 0),
                    ("sa", "va", "kg", 2, "c", 0),
                ],
                names=[
                    "scenario",
                    "variable",
                    "unit",
                    "run_id",
                    "new_variable",
                    "another_new_variable",
                ],
            ),
            id="add-multiple-new-levels",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 5),
                    ("sb", "vb", "m", 1),
                    ("sa", "va", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            {"scenario": ["a", "b", "c"], "extra_variable": [1, 1, 1]},
            pd.MultiIndex.from_tuples(
                [
                    ("a", "va", "kg", 5, 1),
                    ("b", "vb", "m", 1, 1),
                    ("c", "va", "kg", 2, 1),
                ],
                names=["scenario", "variable", "unit", "run_id", "extra_variable"],
            ),
            id="replace-existing-level-and-add-one",
        ),
    ),
)
def test_set_levels(start, levels_to_set, exp):
    res = set_levels(start, levels_to_set=levels_to_set)

    pd.testing.assert_index_equal(res, exp)


def test_set_levels_with_a_dataframe():
    start = pd.MultiIndex.from_tuples(
        [
            ("sa", "va", "kg", 0),
            ("sb", "vb", "m", 1),
            ("sa", "va", "kg", 2),
            ("sa", "vb", "kg", -2),
        ],
        names=["scenario", "variable", "unit", "run_id"],
    )
    start_df = pd.DataFrame(
        np.zeros((start.shape[0], 3)), columns=[2010, 2020, 2030], index=start
    )

    res = set_index_levels(start_df, levels_to_set={"new_variable": "test"})

    exp = pd.MultiIndex.from_tuples(
        [
            ("sa", "va", "kg", 0, "test"),
            ("sb", "vb", "m", 1, "test"),
            ("sa", "va", "kg", 2, "test"),
            ("sa", "vb", "kg", -2, "test"),
        ],
        names=["scenario", "variable", "unit", "run_id", "new_variable"],
    )

    pd.testing.assert_index_equal(res.index, exp)


def test_set_levels_raises_type_error():
    start = pd.Index(range(10), name="index")

    levels_to_set = {"new_variable": "test"}

    with pytest.raises(TypeError):
        set_levels(start, levels_to_set=levels_to_set)


def test_set_levels_raises_value_error():
    start = pd.MultiIndex.from_tuples(
        [
            ("sa", "va", "kg", 0),
            ("sb", "vb", "m", 1),
            ("sa", "va", "kg", 2),
        ],
        names=["scenario", "variable", "unit", "run_id"],
    )

    levels_to_set = {"new_variable": ["a", "b", "c", "d"]}

    with pytest.raises(
        ValueError,
        match="Length of values for level 'new_variable' "
        "does not match index length: 4 != 3",
    ):
        set_levels(start, levels_to_set=levels_to_set)
