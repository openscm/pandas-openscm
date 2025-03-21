"""
Tests of `pandas_openscm.plotting` and `pd.DataFrame.openscm.plot*`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.plotting import plot_plume
from pandas_openscm.testing import create_test_df

plt = pytest.importorskip("matplotlib.pyplot")
pytest_regressions = pytest.importorskip("pytest_regressions")

# Different values of:
# - quantile_over, both str and list[str]
# - quantiles_plumes, both plumes and lines

# - check result

# - units
#   - time units have to be passed by the user
#   - value units can be inferred
#      - if matplotlib recognises the units, let it just do its thing
#      - if matplotlib doesn't, if there is only one unit,
#        the units should appear in the y-axis by default (but can be turned off)

# - palette and dashes handling

# - turning off legend possible
# - legend creation injectable
# - add legend items to existing legend rather than creating fresh

# - ax can be auto-created


def check_plots(
    plot_kwargs: dict[str, Any],
    df: pd.DataFrame,
    image_regression: pytest_regressions.image_regression.ImageRegressionFixture,
    tmp_path: Path,
) -> None:
    fig, ax = plt.subplots()

    plot_plume(df, ax=ax, **plot_kwargs)

    out_file = tmp_path / "fig.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Check this works via the accessor too
    fig, ax = plt.subplots()
    df.openscm.plot_plume(ax=ax, **plot_kwargs)

    out_file = tmp_path / "fig-accessor.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


def test_plot_plume_default(tmp_path, image_regression, setup_pandas_accessor):
    df = create_test_df(
        variables=(("variable_1", "K"),),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=82747),
    )

    check_plots(
        df=df, plot_kwargs={}, image_regression=image_regression, tmp_path=tmp_path
    )


@pytest.mark.parametrize(
    "quantile_over, hue_var, style_var",
    (
        ("scenario", "run", "variable"),
        # Having no style var is definitely trickier
        (["scenario", "variable", "unit"], "run", None),
    ),
)
def test_plot_plume_quantile_over(
    quantile_over, hue_var, style_var, tmp_path, image_regression, setup_pandas_accessor
):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "W")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=56461),
    )

    plot_kwargs = dict(
        quantile_over=quantile_over, hue_var=hue_var, style_var=style_var
    )

    check_plots(
        df=df,
        plot_kwargs=plot_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


def test_plot_plume_pre_calculated():
    assert False


def test_plot_plume_units():
    # Plot two different sets of data with different units
    # Make sure that the unit handling passes through
    assert False
