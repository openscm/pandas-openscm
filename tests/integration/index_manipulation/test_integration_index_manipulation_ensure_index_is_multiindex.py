"""
Test `pandas_openscm.index_manipulation.ensure_index_is_multiindex`
"""

from __future__ import annotations

import numpy as np

from pandas_openscm.index_manipulation import convert_index_to_category_index
from pandas_openscm.testing import create_test_df


def ensure_index_is_multiindex():
    raise NotImplementedError
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


def test_accessor(setup_pandas_accessor):
    raise NotImplementedError
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
