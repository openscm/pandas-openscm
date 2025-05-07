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
    ),
)
def test_update_levels_from_other(start, levels_to_set, exp):
    res = set_levels(start, levels_to_set=levels_to_set)

    pd.testing.assert_index_equal(res, exp)
