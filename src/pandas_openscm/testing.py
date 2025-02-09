"""
Testing helpers

Placed here to avoid putting `__init__.py` files in our `tests` directory,
see details here: https://docs.pytest.org/en/stable/explanation/goodpractices.html#which-import-mode.
Also see here: https://docs.pytest.org/en/stable/explanation/pythonpath.html#pytest-import-mechanisms-and-sys-path-pythonpath.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pandas_openscm.db import OpenSCMDBFormat
from pandas_openscm.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import pytest


def get_parametrized_db_formats() -> pytest.MarkDecorator:
    try:
        import pytest
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_parametrized_db_formats", requirement="pytest"
        ) from exc

    return pytest.mark.parametrize(
        "db_format",
        tuple(
            pytest.param(db_format, id=str(db_format)) for db_format in OpenSCMDBFormat
        ),
    )


def create_test_df(
    *,
    variables: Iterable[tuple[str, str]],
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
