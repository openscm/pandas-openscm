"""
Test `pandas_openscm.index_manipulation.update_index_levels`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.index_manipulation import update_index_levels


@pytest.mark.parametrize(
    "start, updates, exp",
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
            {},
            id="no-changes",
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
            {"variable": lambda x: x.replace("v", "vv")},
            id="single-update",
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
            {
                "variable": lambda x: x.replace("v", "vv"),
                "unit": lambda x: x.replace("kg", "g").replace("m", "km"),
            },
            id="multiple-updates",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [
                    ("sa", "va", "kg", 0),
                    ("sb", "vb", "m", -1),
                    ("sa", "va", "kg", -2),
                    ("sa", "vb", "kg", 2),
                ],
                names=["scenario", "variable", "unit", "run_id"],
            ),
            {
                "variable": lambda x: x.replace("v", "vv"),
                "unit": lambda x: x.replace("kg", "g").replace("m", "km"),
                "run_id": np.abs,
            },
            id="multiple-updates-incl-external-func",
        ),
    ),
)
def test_update_index_levels(start, updates):
    res = update_index_levels(start, updates=updates)

    exp = res.copy()
    for level, func in updates.items():
        exp = exp.set_levels(exp.get_level_values(level).map(func))

    pd.testing.assert_index_equal(res, exp)


def test_update_index_levels_missing_level():
    start = pd.MultiIndex.from_tuples(
        [
            ("sa", "va", "kg", 0),
            ("sb", "vb", "m", -1),
            ("sa", "va", "kg", -2),
            ("sa", "vb", "kg", 2),
        ],
        names=["scenario", "variable", "unit", "run_id"],
    )
    updates = {
        "variable": lambda x: x.replace("v", "vv"),
        "units": lambda x: x.replace("kg", "g").replace("m", "km"),
    }

    update_index_levels(start, updates=updates)
