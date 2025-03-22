"""
Tests of `pandas_openscm.plotting` and `pd.DataFrame.openscm.plot*`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.plotting import (
    get_quantiles,
    plot_plume,
    plot_plume_after_calculating_quantiles,
)
from pandas_openscm.testing import create_test_df

plt = pytest.importorskip("matplotlib.pyplot")
pytest_regressions = pytest.importorskip("pytest_regressions")

# Different values of:
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

# - test error handling if you don't have the quantiles
# - test error handling if quantile over and hue don't line up


def check_plots(
    plot_kwargs: dict[str, Any],
    df: pd.DataFrame,
    image_regression: pytest_regressions.image_regression.ImageRegressionFixture,
    tmp_path: Path,
) -> None:
    fig, ax = plt.subplots()

    return_val = plot_plume(df, ax=ax, **plot_kwargs)

    assert return_val == ax

    out_file = tmp_path / "fig.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Check this works via the accessor too
    fig, ax = plt.subplots()
    return_val = df.openscm.plot_plume(ax=ax, **plot_kwargs)

    assert return_val == ax

    out_file = tmp_path / "fig-accessor.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


def check_plots_incl_quantile_calculation(
    method_kwargs: dict[str, Any],
    df: pd.DataFrame,
    image_regression: pytest_regressions.image_regression.ImageRegressionFixture,
    tmp_path: Path,
) -> None:
    fig, ax = plt.subplots()

    return_val = plot_plume_after_calculating_quantiles(df, ax=ax, **method_kwargs)

    assert return_val == ax

    out_file = tmp_path / "fig.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Check this works via the accessor too
    fig, ax = plt.subplots()
    return_val = df.openscm.plot_plume_after_calculating_quantiles(
        ax=ax, **method_kwargs
    )
    assert return_val == ax

    out_file = tmp_path / "fig-accessor.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


def test_plot_plume_default(tmp_path, image_regression, setup_pandas_accessor):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=82747),
    )

    check_plots(
        df=df.openscm.groupby_except("run")
        .quantile([0.05, 0.5, 0.95])
        .openscm.fix_index_name_after_groupby_quantile(),
        plot_kwargs={},
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize(
    "quantiles_plumes",
    (
        pytest.param(
            (
                (0.5, 0.7),
                ((0.25, 0.75), 0.5),
                ((0.05, 0.95), 0.2),
            ),
            id="multi-plume",
        ),
        pytest.param(
            (
                ((0.25, 0.75), 0.5),
                ((0.05, 0.95), 0.2),
            ),
            id="plumes-only",
        ),
        pytest.param(
            ((0.5, 0.7),),
            id="line-only",
        ),
        # If you actually wanted to do this,
        # you would just use the seaborn API directly,
        # but this at least checks that things don't explode.
        pytest.param(
            (
                (0.5, 0.7),
                (0.05, 0.7),
                (0.95, 0.7),
            ),
            id="lines-only",
        ),
    ),
)
def test_plot_plume_quantiles(
    quantiles_plumes, tmp_path, image_regression, setup_pandas_accessor
):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=82747),
    )

    plot_kwargs = dict(quantiles_plumes=quantiles_plumes, linewidth=1)

    check_plots(
        df=df.openscm.groupby_except("run")
        .quantile(get_quantiles(quantiles_plumes))
        .openscm.fix_index_name_after_groupby_quantile(),
        plot_kwargs=plot_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize(
    "quantile_over, hue_var, style_var, kwargs",
    (
        pytest.param(
            "scenario",
            "run",
            "variable",
            dict(warn_infer_y_label_with_multi_unit=False),
            id="single-var-with-style-var",
        ),
        pytest.param(
            ["scenario", "variable", "unit"],
            "run",
            None,
            dict(unit_col=None),
            id="multi-var-with-no-style-var",
        ),
    ),
)
def test_plot_plume_quantile_over(  # noqa: PLR0913
    quantile_over,
    hue_var,
    style_var,
    kwargs,
    tmp_path,
    image_regression,
    setup_pandas_accessor,
):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "W")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=56461),
    )

    method_kwargs = dict(
        quantile_over=quantile_over,
        quantiles_plumes=((0.5, 0.5), ((0.05, 0.95), 0.2)),
        hue_var=hue_var,
        style_var=style_var,
        **kwargs,
    )

    check_plots_incl_quantile_calculation(
        df=df,
        method_kwargs=method_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


# def test_plot_plume_units():
#     # Plot two different sets of data with different units
#     # Make sure that the unit handling passes through
#     assert False
