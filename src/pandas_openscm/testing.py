"""
Testing helpers

Placed here to avoid putting `__init__.py` files in our `tests` directory,
see details here: https://docs.pytest.org/en/stable/explanation/goodpractices.html#which-import-mode.
Also see here: https://docs.pytest.org/en/stable/explanation/pythonpath.html#pytest-import-mechanisms-and-sys-path-pythonpath.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pandas_openscm.db import CSVBackend, FeatherBackend, MovePlan, netCDFBackend
from pandas_openscm.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import pytest


def get_parametrized_db_backends() -> pytest.MarkDecorator:
    try:
        import pytest
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_parametrized_db_backends", requirement="pytest"
        ) from exc

    return pytest.mark.parametrize(
        "db_backend",
        tuple(
            pytest.param(db_format, id=str(db_format))
            for db_format in (
                CSVBackend,
                FeatherBackend,
                netCDFBackend,
                # Other back-end options to consider:
                #
                # pretty netCDF, where we try and save the data with dimensions
                # where possible
                #
                # HDF5: https://pandas.pydata.org/docs/user_guide/io.html#hdf5-pytables
                # HDF5 = auto()
            )
        ),
    )


def assert_frame_alike(
    res: pd.DataFrame, exp: pd.DataFrame, check_like: bool = True, **kwargs: Any
) -> None:
    """
    Assert that two [pd.DataFrame][`pandas.DataFrame`] are alike

    Here, alike means that they have the same data,
    just potentially not in the same order.
    This includes the order of index levels, which may also differ.

    Parameters
    ----------
    res
        Result to check

    exp
        Expected result

    check_like
        Passed to [`assert_frame_equal`][pandas.testing.assert_frame_equal]

    **kwargs
        Passed to [`assert_frame_equal`][pandas.testing.assert_frame_equal]
    """
    pd.testing.assert_frame_equal(
        res.reorder_levels(exp.index.names),
        exp,
        check_like=check_like,
        **kwargs,
    )


def create_test_df(
    *,
    variables: Collection[tuple[str, str]],
    n_scenarios: int,
    n_runs: int,
    timepoints: np.typing.NDArray[np.floating],
) -> pd.DataFrame:
    """
    Create a [`pd.DataFrame`][pandas.DataFrame] to use in testing

    This uses the idea of simple climate model runs,
    where you have a number of scenarios,
    each of which has a number of variables
    from a number of different model runs
    with output for a number of different time points.

    The result will contain all combinations of scenarios,
    variables and runs,
    with the units being defined by each variable.

    Parameters
    ----------
    variables
        Variables and their units to create

    n_scenarios
        Number of scenarios to create.

        These are simply incremented with their number.

    n_runs
        Number of runs to create.

        These are simply numbered.

    timepoints
        Time points to use with the data.

    Returns
    -------
    :
        Generated test [`pd.DataFrame`][pandas.DataFrame].
    """
    idx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            (
                (s, v_info[0], r, v_info[1])
                for s, v_info, r in itertools.product(
                    [f"scenario_{i}" for i in range(n_scenarios)],
                    variables,
                    [i for i in range(n_runs)],
                )
            ),
            columns=["scenario", "variable", "run", "unit"],
        )
    )

    n_variables = len(variables)
    n_ts = n_scenarios * n_variables * n_runs

    # Give the data a bit of structure so it looks different when plotted.
    values = 50.0 * np.linspace(0.3, 1, n_ts)[:, np.newaxis] * np.linspace(
        0, 1, timepoints.size
    )[np.newaxis, :] + np.random.default_rng().random((n_ts, timepoints.size))

    df = pd.DataFrame(
        values,
        columns=timepoints,
        index=idx,
    )

    return df


def assert_move_plan_equal(res: MovePlan, exp: MovePlan) -> None:
    """
    Assert that two [`MovePlan`][(p).db.MovePlan] are equal

    Parameters
    ----------
    res
        The result

    exp
        The expectation

    Raises
    ------
    AssertionError
        `res` and `exp` are not equal
    """
    # Check that the indexes are the same.
    # We convert to MultiIndex first as we don't care about the actual index values.
    pd.testing.assert_index_equal(
        pd.MultiIndex.from_frame(res.moved_index.reset_index()),
        pd.MultiIndex.from_frame(exp.moved_index.reset_index()),
        check_order=False,
    )
    pd.testing.assert_series_equal(
        res.moved_file_map, exp.moved_file_map, check_like=True
    )

    if res.rewrite_actions is None:
        assert exp.rewrite_actions is None
    else:
        if exp.rewrite_actions is None:
            msg = f"{exp.rewrite_actions=} while {res.rewrite_actions=}"
            raise AssertionError(msg)

        assert len(res.rewrite_actions) == len(exp.rewrite_actions)
        for res_rwa in res.rewrite_actions:
            for exp_rwa in exp.rewrite_actions:
                if res_rwa.from_file == exp_rwa.from_file:
                    break
            else:
                msg = f"Did not find pair for\n{res_rwa=}\nin\n{exp.rewrite_actions=}"
                raise AssertionError(msg)

            pd.testing.assert_index_equal(
                res_rwa.locator, exp_rwa.locator, check_order=False
            )
            assert res_rwa.to_file == exp_rwa.to_file

    if res.delete_paths is None:
        assert exp.delete_paths is None
    else:
        if exp.delete_paths is None:
            msg = f"{exp.delete_paths=} while {res.delete_paths=}"
            raise AssertionError(msg)

        assert set(res.delete_paths) == set(exp.delete_paths)
