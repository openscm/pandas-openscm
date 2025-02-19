"""
Test `pandas_openscm.index_manipulation.convert_index_to_category_index`
"""

from __future__ import annotations

import numpy as np

from pandas_openscm.index_manipulation import convert_index_to_category_index
from pandas_openscm.testing import create_test_df


def run_checks(res, start):
    # Check that columns are now all category types
    for idx_lvl in res.index.names:
        # Check that we didn't start with categories or mangle the original DataFrame
        assert (
            str(start.index.get_level_values(idx_lvl).dtype) != "category"
        ), "Testing nothing"

        assert str(res.index.get_level_values(idx_lvl).dtype) == "category"

    # Check that memory usage went down.
    # Unclear to me why this doesn't work if I try and use memory_usage
    # on the index directly, without casting to frame first.
    assert (
        res.index.to_frame(index=False).memory_usage().sum()
        / start.index.to_frame(index=False).memory_usage().sum()
    ) < 0.5


def test_to_category_index():
    units = ["Mt", "kg", "W"]

    # Biggish DataFrame
    start = create_test_df(
        variables=[(f"variable_{i}", units[i % len(units)]) for i in range(25)],
        n_scenarios=30,
        n_runs=60,
        timepoints=np.arange(1750.0, 2100.0 + 1.0),
    )

    res = convert_index_to_category_index(start)

    run_checks(res, start)


def test_to_category_index_series():
    units = ["Mt", "kg", "W"]

    # Biggish DataFrame
    start = create_test_df(
        variables=[(f"variable_{i}", units[i % len(units)]) for i in range(25)],
        n_scenarios=30,
        n_runs=60,
        timepoints=np.arange(1750.0, 1752.0),
    )[1751.0]

    res = convert_index_to_category_index(start)

    run_checks(res, start)


def test_accessor(setup_pandas_accessor):
    units = ["Mt", "kg", "W"]

    # Biggish DataFrame
    start = create_test_df(
        variables=[(f"variable_{i}", units[i % len(units)]) for i in range(25)],
        n_scenarios=30,
        n_runs=60,
        timepoints=np.arange(1750.0, 2100.0 + 1.0),
    )

    res = start.openscm.to_category_index()

    run_checks(res, start)
