"""
Test `pandas_openscm.index_manipulation.set_levels`
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas_openscm.index_manipulation import set_levels


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
            id="set-multiple-levels",
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
            {"variable": ["a", "b", "c"]},
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "a", "kg", 0),
                    ("sb", "b", "m", 1),
                    ("sa", "c", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            id="alter-existing-level",
        ),
    ),
)
def test_set_levels(start, levels_to_set, exp):
    res = set_levels(start, levels_to_set=levels_to_set)

    pd.testing.assert_index_equal(res, exp)


def test_set_levels_raises_type_error():
    """
    Test that set_levels raises an error when the levels to set are not in the index.
    """
    start = pd.Index(range(10), name="index")

    levels_to_set = {"new_variable": "test"}

    with pytest.raises(TypeError):
        set_levels(start, levels_to_set=levels_to_set)


def test_set_levels_raises_value_error():
    """
    Test that set_levels raises an error when the levels to set are same length
    as index.
    """
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
        ValueError, match="Length of values does not match, got 4 but expected 3"
    ):
        set_levels(start, levels_to_set=levels_to_set)
