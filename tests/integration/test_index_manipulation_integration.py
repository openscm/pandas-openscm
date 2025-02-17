"""
Tests of `pandas_openscm.index_manipulation`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandas_openscm.index_manipulation import unify_index_levels


def assert_index_equal_here(res: pd.MultiIndex, exp: pd.MultiIndex):
    """
    Assert that indexes are equal

    We make our life a bit easier by removing unused levels from `res` before comparing.
    """
    pd.testing.assert_index_equal(res.remove_unused_levels(), exp)


def test_unify_index_levels_already_matching():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2, 3),
            (4, 5, 6),
        ],
        names=["a", "b", "c"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (7, 8, 9),
            (10, 11, 12),
        ],
        names=["a", "b", "c"],
    )

    # Should be no change
    exp_a = idx_a
    # Should be no change
    exp_b = idx_b

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_just_reordering():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2, 3),
            (4, 5, 6),
        ],
        names=["a", "b", "c"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (7, 8, 9),
            (10, 11, 12),
        ],
        names=["c", "a", "b"],
    )

    # Should be no change
    exp_a = idx_a
    exp_b = idx_b.reorder_levels(idx_a.names)

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_a_within_b():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2),
            (4, 5),
        ],
        names=["a", "b"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (7, 8, 9),
            (10, 11, 12),
        ],
        names=["a", "b", "c"],
    )

    exp_a = pd.MultiIndex(
        levels=[[1, 4], [2, 5], np.array([], dtype=np.int64)],
        codes=[[0, 1], [0, 1], [-1, -1]],
        names=["a", "b", "c"],
    )

    # Should be no change
    exp_b = idx_b

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_a_within_b_skip_a_level():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2),
            (4, 5),
        ],
        names=["a", "c"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (7, 8, 9),
            (10, 11, 12),
        ],
        names=["a", "b", "c"],
    )

    exp_a = pd.MultiIndex(
        levels=[[1, 4], [2, 5], np.array([], dtype=np.int64)],
        codes=[[0, 1], [0, 1], [-1, -1]],
        names=["a", "c", "b"],
    )

    # unify_index_levels is not 100% symmetric
    # TODO: put in docstring
    exp_b = idx_b.reorder_levels(["a", "c", "b"])

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_a_outside_b():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2, 3),
            (4, 5, 6),
        ],
        names=["a", "b", "c"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (7, 8),
            (10, 11),
        ],
        names=["a", "b"],
    )

    exp_b = pd.MultiIndex(
        levels=[[7, 10], [8, 11], np.array([], dtype=np.int64)],
        codes=[[0, 1], [0, 1], [-1, -1]],
        names=["a", "b", "c"],
    )

    # Should be no change
    exp_a = idx_a

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_a_outside_b_skip_level():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2, 3),
            (4, 5, 6),
        ],
        names=["a", "b", "c"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (7, 8),
            (10, 11),
        ],
        names=["b", "c"],
    )

    exp_b = pd.MultiIndex(
        levels=[np.array([], dtype=np.int64), [7, 10], [8, 11]],
        codes=[[-1, -1], [0, 1], [0, 1]],
        names=["a", "b", "c"],
    )

    # Should be no change
    exp_a = idx_a

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)
