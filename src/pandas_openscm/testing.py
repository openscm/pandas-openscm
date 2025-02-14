"""
Testing helpers

Placed here to avoid putting `__init__.py` files in our `tests` directory,
see details here: https://docs.pytest.org/en/stable/explanation/goodpractices.html#which-import-mode.
Also see here: https://docs.pytest.org/en/stable/explanation/pythonpath.html#pytest-import-mechanisms-and-sys-path-pythonpath.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection
from typing import TYPE_CHECKING

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


def assert_move_plan_equal(a: MovePlan, b: MovePlan) -> None:
    """
    Assert that two [`MovePlan`][(p).db.MovePlan] are equal

    Parameters
    ----------
    a
        First object to check

    b
        Other object to check

    Raises
    ------
    AssertionError
        `a` and `b` are not equal
    """
    # Check that the indexes are the same.
    # We convert to MultiIndex first as we don't care about the actual index values.
    pd.testing.assert_index_equal(
        pd.MultiIndex.from_frame(a.moved_index),
        pd.MultiIndex.from_frame(b.moved_index),
        check_order=False,
    )
    pd.testing.assert_series_equal(a.moved_file_map, b.moved_file_map, check_like=True)

    if a.rewrite_actions is None:
        assert b.rewrite_actions is None
    else:
        if b.rewrite_actions is None:
            msg = f"{b.rewrite_actions=} while {a.rewrite_actions=}"
            raise AssertionError(msg)

        assert len(a.rewrite_actions) == len(b.rewrite_actions)
        for ara in a.rewrite_actions:
            for bra in b.rewrite_actions:
                if ara.from_file == bra.from_file:
                    break
            else:
                msg = f"Did not find pair for\n{ara=}\nin\n{b.rewrite_actions=}"
                raise AssertionError(msg)

            pd.testing.assert_index_equal(ara.locator, bra.locator, check_order=False)
            assert ara.to_file == bra.to_file

    if a.delete_paths is None:
        assert b.delete_paths is None
    else:
        if b.delete_paths is None:
            msg = f"{b.delete_paths=} while {a.delete_paths=}"
            raise AssertionError(msg)

        assert set(a.delete_paths) == set(b.delete_paths)
