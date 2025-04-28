"""
Test `pandas_openscm.index_manipulation.update_levels_from_other`
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.index_manipulation import (
    update_index_levels_from_other_func,
    update_levels_from_other,
)


@pytest.mark.parametrize(
    "start, update_sources",
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
            {"variable": ("variable", lambda x: x.replace("v", "vv"))},
            # Slightly stupid test as you would just use update but ok
            id="single-update-of-self",
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
            {"vv": ("variable", lambda x: x.replace("v", "vv"))},
            id="single-new-variable",
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
                "vv": ("variable", lambda x: x.replace("v", "vv")),
                "uu": ("unit", lambda x: x.replace("kg", "g").replace("m", "km")),
            },
            id="multiple-new-variables",
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
            {"scenario": ("variable", lambda x: x.replace("v", "vv"))},
            id="single-updated-variable",
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
                "scenario": ("variable", lambda x: x.replace("v", "vv")),
                "model": ("unit", lambda x: x.replace("kg", "g").replace("m", "km")),
            },
            id="multiple-updated-variables",
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
                "scenario": ("variable", lambda x: x.replace("v", "vv")),
                "unit": ("unit", lambda x: x.replace("kg", "g").replace("m", "km")),
            },
            id="updated-and-new-variable",
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
            {"v_new": ("variable", lambda x: x[0])},
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
                "vv": ("variable", lambda x: x.replace("v", "vv")),
                "unit": ("unit", lambda x: x.replace("kg", "g").replace("m", "km")),
                "run_id_abs": ("run_id", np.abs),
            },
            id="multiple-updates-incl-external-func",
        ),
    ),
)
def test_update_levels_from_other(start, update_sources):
    res = update_levels_from_other(start, update_sources=update_sources)

    exp = start.to_frame(index=False)
    for level, (source, mapper) in update_sources.items():
        exp[level] = exp[source].map(mapper)
    exp = pd.MultiIndex.from_frame(exp)

    pd.testing.assert_index_equal(res, exp)


def test_update_levels_from_other_missing_level():
    start = pd.MultiIndex.from_tuples(
        [
            ("sa", "va", "kg", 0),
            ("sb", "vb", "m", -1),
            ("sa", "va", "kg", -2),
            ("sa", "vb", "kg", 2),
        ],
        names=["scenario", "variable", "unit", "run_id"],
    )
    update_sources = {
        "vv": ("variable", lambda x: x.replace("v", "vv")),
        "uu": ("units", lambda x: x.replace("kg", "g").replace("m", "km")),
    }

    with pytest.raises(
        KeyError,
        match=re.escape(
            "units is not available in the index. "
            f"Available levels: {['scenario', 'variable', 'unit', 'run_id']}"
        ),
    ):
        update_levels_from_other(start, update_sources=update_sources)


def test_doesnt_trip_over_droped_levels(setup_pandas_accessor):
    def update_func(in_v: int) -> int:
        if in_v < 0:
            msg = f"Value must be greater than zero, received {in_v}"
            raise ValueError(msg)

        return in_v * -1

    start = pd.MultiIndex.from_tuples(
        [
            ("sa", "va", "kg", 0),
            ("sb", "vb", "m", 1),
            ("sa", "va", "kg", 2),
            ("sa", "vb", "kg", -2),
        ],
        names=["scenario", "variable", "unit", "run_id"],
    )

    update_sources = {"run_id": ("run_id", update_func)}

    res = update_levels_from_other(start[:-1], update_sources=update_sources)

    exp = pd.MultiIndex.from_tuples(
        [
            ("sa", "va", "kg", 0),
            ("sb", "vb", "m", -1),
            ("sa", "va", "kg", -2),
        ],
        names=["scenario", "variable", "unit", "run_id"],
    )
    pd.testing.assert_index_equal(res, exp)

    # If you turn the drop off, you get an error
    exp_error_no_removal = pytest.raises(
        ValueError, match=re.escape("Value must be greater than zero, received -2")
    )
    with exp_error_no_removal:
        # Even though we're not using the levels,
        # they still get mapped if we don't remove them
        update_levels_from_other(
            start[:-1], update_sources=update_sources, remove_unused_levels=False
        )

    # Same thing but from a DataFrame
    start_df = pd.DataFrame(
        np.zeros((start.shape[0], 3)), columns=[2010, 2020, 2030], index=start
    )

    res_df = update_index_levels_from_other_func(
        start_df.iloc[:-1, :], update_sources=update_sources
    )

    exp_df = pd.DataFrame(
        np.zeros((exp.shape[0], 3)), columns=start_df.columns, index=exp
    )

    pd.testing.assert_frame_equal(res_df, exp_df)
    with exp_error_no_removal:
        update_index_levels_from_other_func(
            start_df.iloc[:-1, :],
            update_sources=update_sources,
            remove_unused_levels=False,
        )

    # Lastly, test the accessor
    pd.testing.assert_frame_equal(
        start_df.iloc[:-1, :].openscm.update_index_levels_from_other(update_sources),
        exp_df,
    )
    with exp_error_no_removal:
        start_df.iloc[:-1, :].openscm.update_index_levels_from_other(
            update_sources, remove_unused_levels=False
        )


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

    update_sources = {
        "vv": ("variable", lambda x: x.replace("v", "vv")),
        "unit": ("unit", lambda x: x.replace("kg", "g").replace("m", "km")),
    }

    exp = pd.DataFrame(
        start.values,
        columns=start.columns,
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "va", "g", 0, "vva"),
                ("sb", "vb", "km", -1, "vvb"),
                ("sa", "va", "g", -2, "vva"),
                ("sa", "vb", "g", 2, "vvb"),
            ],
            names=["scenario", "variable", "unit", "run_id", "vv"],
        ),
    )

    res = start.openscm.update_index_levels_from_other(update_sources)
    pd.testing.assert_frame_equal(res, exp)

    # Test function too
    res = update_index_levels_from_other_func(start, update_sources)
    pd.testing.assert_frame_equal(res, exp)


def test_accessor_not_multiindex(setup_pandas_accessor):
    start = pd.DataFrame(np.arange(2 * 4).reshape((4, 2)))

    error_msg = re.escape(
        "This function is only intended to be used "
        "when `df`'s index is an instance of `MultiIndex`. "
        "Received type(df.index)="
    )
    with pytest.raises(TypeError, match=error_msg):
        start.openscm.update_index_levels_from_other({})

    with pytest.raises(TypeError, match=error_msg):
        update_index_levels_from_other_func(start, {})
