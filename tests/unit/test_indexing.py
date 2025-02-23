"""
Basic unit tests of `pandas_openscm.indexing`
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from pandas_openscm.indexing import (
    index_name_aware_lookup,
    multi_index_lookup,
    multi_index_match,
)


def test_unusable_index_levels():
    idx = pd.MultiIndex.from_tuples(
        [
            ("a", "b"),
            ("c", "d"),
        ],
        names=["model", "scenario"],
    )

    locator = pd.MultiIndex.from_tuples(
        [
            ("a", "b", 1),
            ("c", "d", 2),
        ],
        names=["model", "scenario", "run_id"],
    )

    with pytest.raises(
        KeyError,
        match=re.escape(
            "The following levels in `locator` are not in `idx`: ['run_id']. "
            f"{locator.names=} {idx.names=}"
        ),
    ):
        multi_index_match(idx=idx, locator=locator)


def test_index_is_not_multi_index():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "This function is only intended to be used "
            "when `df`'s index is an instance of `MultiIndex`. "
            "Received type(pandas_obj.index)"
            "=<class 'pandas.core.indexes.range.RangeIndex'>"
        ),
    ):
        multi_index_lookup(pd.DataFrame(pd.Index([0, 1])), locator="not used")


def test_index_name_aware_lookup():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "This function is only intended to be used "
            "when `df`'s index is an instance of `MultiIndex`. "
            "Received type(pandas_obj.index)"
            "=<class 'pandas.core.indexes.range.RangeIndex'>"
        ),
    ):
        index_name_aware_lookup(pd.DataFrame(pd.Index([0, 1])), locator="not used")
