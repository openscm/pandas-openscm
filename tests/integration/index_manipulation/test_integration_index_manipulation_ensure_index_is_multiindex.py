"""
Test `pandas_openscm.index_manipulation.ensure_index_is_multiindex`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.index_manipulation import (
    ensure_index_is_multiindex,
    ensure_is_multiindex,
)
from pandas_openscm.testing import (
    convert_to_desired_type,
    create_test_df,
)


@pytest.mark.parametrize("copy, copy_exp", ((None, True), (True, True), (False, False)))
def test_ensure_index_is_multiindex(copy, copy_exp):
    start = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=[10, 20],
        index=pd.Index(["a", "b"], name="variable"),
    )

    call_kwargs = {}
    if copy is not None:
        call_kwargs["copy"] = copy

    res = ensure_index_is_multiindex(start, **call_kwargs)

    assert isinstance(res.index, pd.MultiIndex)
    assert res.index.names == ["variable"]

    if copy_exp:
        # New object returned
        assert id(start) != id(res)

        assert isinstance(start.index, pd.Index)

    else:
        # Same object returned
        assert id(start) == id(res)
        # Therefore affects input object too
        assert isinstance(start.index, pd.MultiIndex)


@pytest.mark.parametrize("copy, copy_exp", ((None, True), (True, True), (False, False)))
def test_ensure_index_is_multiindex_no_op(copy, copy_exp):
    start = create_test_df(
        variables=[("Temperature", "K")],
        n_scenarios=2,
        n_runs=2,
        timepoints=np.arange(1750.0, 2100.0 + 1.0),
    )

    call_kwargs = {}
    if copy is not None:
        call_kwargs["copy"] = copy

    res = ensure_index_is_multiindex(start, **call_kwargs)

    # Already a MultiIndex, should be no change
    pd.testing.assert_index_equal(res.index, start.index)

    # Behaviour of copy should be respected to avoid confusing behaviour
    if copy_exp:
        # New object returned
        assert id(start) != id(res)

    else:
        # Same object returned
        assert id(start) == id(res)


@pytest.mark.parametrize("copy, copy_exp", ((None, True), (True, True), (False, False)))
@pytest.mark.parametrize(
    "pobj_type",
    ("DataFrame", "Series"),
)
def test_accessor(setup_pandas_accessors, copy, copy_exp, pobj_type):
    start = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=[10, 20],
        index=pd.Index(["a", "b"], name="variable"),
    )
    start = convert_to_desired_type(start, pobj_type)

    call_kwargs = {}
    if copy is not None:
        call_kwargs["copy"] = copy

    res = start.openscm.ensure_index_is_multiindex(**call_kwargs)

    assert isinstance(res.index, pd.MultiIndex)
    assert res.index.names == ["variable"]
    if copy_exp:
        # New object returned
        assert id(start) != id(res)

    else:
        # Same object returned
        assert id(start) == id(res)

    # Test alias too
    res_short = start.openscm.eiim(**call_kwargs)

    assert isinstance(res_short.index, pd.MultiIndex)
    assert res_short.index.names == ["variable"]
    if copy_exp:
        # New object returned
        assert id(start) != id(res_short)

    else:
        # Same object returned
        assert id(start) == id(res_short)


def test_ensure_is_multiindex_index():
    start = pd.Index([1, 2, 3], name="id")

    res = ensure_is_multiindex(start)

    assert isinstance(res, pd.MultiIndex)

    exp = pd.MultiIndex.from_tuples(
        [
            (1,),
            (2,),
            (3,),
        ],
        names=["id"],
    )

    pd.testing.assert_index_equal(res, exp)


def test_ensure_is_multiindex_index_no_name():
    start = pd.Index([1, 2, 3])

    res = ensure_is_multiindex(start)

    assert isinstance(res, pd.MultiIndex)

    exp = pd.MultiIndex.from_tuples(
        [
            (1,),
            (2,),
            (3,),
        ],
        names=[None],
    )

    pd.testing.assert_index_equal(res, exp)


def test_ensure_is_multiindex():
    start = pd.MultiIndex.from_tuples(
        [
            ("a", "b"),
            ("c", "d"),
        ],
        names=["mod", "scen"],
    )

    res = ensure_is_multiindex(start)

    # Same object returned
    assert id(start) == id(res)
    assert isinstance(res, pd.MultiIndex)
    pd.testing.assert_index_equal(res, start)


def test_ensure_is_multiindex_accessor_index(setup_pandas_accessors):
    start = pd.Index([1, 2, 3], name="id")

    res = start.openscm.ensure_is_multiindex()
    res_short = start.openscm.eim()
    pd.testing.assert_index_equal(res, res_short)

    assert isinstance(res, pd.MultiIndex)

    exp = pd.MultiIndex.from_tuples(
        [
            (1,),
            (2,),
            (3,),
        ],
        names=["id"],
    )

    pd.testing.assert_index_equal(res, exp)


def test_ensure_is_multiindex_accessor_multiindex(setup_pandas_accessors):
    start = pd.MultiIndex.from_tuples(
        [
            ("a", "b"),
            ("c", "d"),
        ],
        names=["mod", "scen"],
    )

    res = start.openscm.ensure_is_multiindex()
    res_short = start.openscm.eim()
    pd.testing.assert_index_equal(res, res_short)

    # Same object returned
    assert id(start) == id(res)
    assert isinstance(res, pd.MultiIndex)
    pd.testing.assert_index_equal(res, start)
