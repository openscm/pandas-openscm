"""
Test `pandas_openscm.index_manipulation.unify_index_levels`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandas_openscm.index_manipulation import unify_index_levels


def assert_index_equal_here(res: pd.MultiIndex, exp: pd.MultiIndex):
    """
    Assert that indexes are equal

    Special function here in case we need to do any pre-processing.
    """
    pd.testing.assert_index_equal(res, exp)


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


def test_unify_index_levels_partial_intersecting():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2, 3),
            (4, 5, 6),
        ],
        names=["a", "b", "d"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (7, 8, 9),
            (10, 11, 12),
        ],
        names=["a", "b", "c"],
    )

    exp_a = pd.MultiIndex(
        levels=[[1, 4], [2, 5], [3, 6], np.array([], dtype=np.int64)],
        codes=[[0, 1], [0, 1], [0, 1], [-1, -1]],
        names=["a", "b", "d", "c"],
    )

    exp_b = pd.MultiIndex(
        levels=[[7, 10], [8, 11], np.array([], dtype=np.int64), [9, 12]],
        codes=[[0, 1], [0, 1], [-1, -1], [0, 1]],
        names=["a", "b", "d", "c"],
    )

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_non_intersecting():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (1, 2),
            (3, 4),
        ],
        names=["a", "c"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (5, 6),
            (7, 8),
        ],
        names=["b", "d"],
    )

    exp_a = pd.MultiIndex(
        levels=[
            [1, 3],
            [2, 4],
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        ],
        codes=[[0, 1], [0, 1], [-1, -1], [-1, -1]],
        names=["a", "c", "b", "d"],
    )

    exp_b = pd.MultiIndex(
        levels=[
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            [5, 7],
            [6, 8],
        ],
        codes=[[-1, -1], [-1, -1], [0, 1], [0, 1]],
        names=["a", "c", "b", "d"],
    )

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_ordering():
    idx_a = pd.MultiIndex.from_tuples(
        [
            (np.nan, 2, 1, np.nan),
        ],
        names=["a", "b", "c", "d"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            (11, 12),
            (5, 4),
            (7, 8),
        ],
        names=["b", "c"],
    )

    # Expect no change
    exp_a = idx_a

    exp_b = pd.MultiIndex(
        levels=[
            [],
            [11, 5, 7],
            [12, 4, 8],
            [],
        ],
        codes=[[-1, -1, -1], [0, 1, 2], [0, 1, 2], [-1, -1, -1]],
        names=["a", "b", "c", "d"],
    )

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)


def test_unify_index_levels_overlap():
    idx_a = pd.MultiIndex.from_tuples(
        [
            ("scenario_a", "model_1", "variable_1"),
            ("scenario_a", "model_2", "variable_1"),
            ("scenario_a", "model_1", "variable_2"),
            ("scenario_a", "model_2", "variable_2"),
            ("scenario_b", "model_1", "variable_1"),
            ("scenario_b", "model_2", "variable_1"),
            ("scenario_b", "model_1", "variable_2"),
            ("scenario_b", "model_2", "variable_2"),
        ],
        names=["scenario", "model", "variable"],
    )

    idx_b = pd.MultiIndex.from_tuples(
        [
            ("scenario_a", "variable_1"),
            ("scenario_a", "variable_1"),
            ("scenario_a", "variable_2"),
            ("scenario_a", "variable_2"),
            ("scenario_b", "variable_1"),
            ("scenario_b", "variable_1"),
            ("scenario_b", "variable_2"),
            ("scenario_b", "variable_2"),
        ],
        names=["scenario", "variable"],
    )

    # Expect no change
    exp_a = idx_a

    exp_b = pd.MultiIndex(
        levels=[
            ["scenario_a", "scenario_b"],
            [],
            ["variable_1", "variable_2"],
        ],
        codes=[
            [0, 0, 0, 0, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 1, 1, 0, 0, 1, 1],
        ],
        names=["scenario", "model", "variable"],
    )

    res_a, res_b = unify_index_levels(idx_a, idx_b)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)

    # Do it the other way around too
    res_b, res_a = unify_index_levels(idx_b, idx_a)

    exp_order = [*idx_b.names, *res_a.names.difference(idx_b.names)]
    exp_a = res_a.reorder_levels(exp_order)
    exp_b = res_b.reorder_levels(exp_order)

    assert_index_equal_here(res_a, exp_a)
    assert_index_equal_here(res_b, exp_b)
