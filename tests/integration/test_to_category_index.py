"""
Test `pd.DataFrame.openscm.to_category_index`
"""

from __future__ import annotations

import numpy as np

from pandas_openscm.accessors import register_pandas_accessor
from pandas_openscm.testing import create_test_df


def test_to_category_index():
    register_pandas_accessor()

    units = ["Mt", "kg", "W"]

    # Biggish DataFrame
    start = create_test_df(
        variables=[(f"variable_{i}", units[i % len(units)]) for i in range(25)],
        n_scenarios=30,
        n_runs=60,
        timepoints=np.arange(1750.0, 2100.0 + 1.0),
    )

    res = start.openscm.to_category_index()

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
