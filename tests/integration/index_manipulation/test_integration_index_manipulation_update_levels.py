"""
Test `pandas_openscm.index_manipulation.update_levels`
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.index_manipulation import update_index_levels_func, update_levels


@pytest.mark.parametrize(
    "start, updates",
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
            {"variable": lambda x: x[0]},
            id="updates-lead-to-dups",
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
    res = update_levels(start, updates=updates)

    exp = start.to_frame(index=False)
    for level, func in updates.items():
        exp[level] = exp[level].map(func)
    exp = pd.MultiIndex.from_frame(exp)

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

    with pytest.raises(
        KeyError,
        match=re.escape(
            "units is not available in the index. "
            f"Available levels: {['scenario', 'variable', 'unit', 'run_id']}"
        ),
    ):
        update_levels(start, updates=updates)


def test_accessor(setup_pandas_accessor):
    start = pd.DataFrame(
        np.arange(2 * 4).reshape((4, 2)),
        columns=[2010, 2020],
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "va", "kg", 0),
                ("sb", "vb", "m", -1),
                ("sa", "va", "kg", -2),
                ("sa", "vb", "kg", 2),
            ],
            names=["scenario", "variable", "unit", "run_id"],
        ),
    )

    updates = {
        "variable": lambda x: x.replace("v", "vv"),
        "unit": lambda x: x.replace("kg", "g").replace("m", "km"),
    }

    exp = pd.DataFrame(
        start.values,
        columns=start.columns,
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "vva", "g", 0),
                ("sb", "vvb", "km", -1),
                ("sa", "vva", "g", -2),
                ("sa", "vvb", "g", 2),
            ],
            names=["scenario", "variable", "unit", "run_id"],
        ),
    )

    res = start.openscm.update_index_levels(updates)
    pd.testing.assert_frame_equal(res, exp)

    # Test function too
    res = update_index_levels_func(start, updates)
    pd.testing.assert_frame_equal(res, exp)
