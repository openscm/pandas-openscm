"""
Tests of `pandas_openscm.grouping` and associated accessors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.grouping import groupby_except
from pandas_openscm.testing import create_test_df


@pytest.mark.parametrize(
    "non_groupers, expected_groups",
    (
        ("run", ["scenario", "variable", "unit"]),
        (["run"], ["scenario", "variable", "unit"]),
        (["run", "scenario"], ["variable", "unit"]),
    ),
)
def test_groupby_except(non_groupers, expected_groups, setup_pandas_accessor):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "W")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(10.0),
    )

    pd.testing.assert_frame_equal(
        # Have to do a calculation after the groupby so there is something to compare
        df.groupby(expected_groups).mean(),
        groupby_except(df, non_groupers).mean(),
    )

    # Also test via the accessor
    pd.testing.assert_frame_equal(
        # Have to do a calculation after the groupby so there is something to compare
        df.groupby(expected_groups).mean(),
        df.openscm.groupby_except(non_groupers).mean(),
    )


@pytest.mark.parametrize(
    "new_name, quantile_exp", ((None, "quantile"), ("percentile", "percentile"))
)
def test_fix_index_name_after_groupby_quantile(
    new_name, quantile_exp, setup_pandas_accessor
):
    fix_kwargs = {}
    if new_name is not None:
        fix_kwargs["new_name"] = new_name

    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "W")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(10.0),
    )

    df_group_raw = df.openscm.groupby_except(["run"]).quantile([0.05, 0.5, 0.95])
    assert df_group_raw.index.names == [
        "scenario",
        "variable",
        "unit",
        None,  # This is the problem we want to fix
    ]

    df_group_fixed = df_group_raw.openscm.fix_index_name_after_groupby_quantile(
        **fix_kwargs
    )
    assert df_group_fixed.index.names == [
        "scenario",
        "variable",
        "unit",
        quantile_exp,  # fixed
    ]
