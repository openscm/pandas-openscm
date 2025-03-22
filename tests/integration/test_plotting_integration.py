"""
Tests of `pandas_openscm.plotting` and `pd.DataFrame.openscm.plot*`
"""

from __future__ import annotations

import contextlib
import re
import sys
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.plotting import (
    get_default_colour_cycler,
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
    exp: contextlib.AbstractContextManager = does_not_raise(),
) -> None:
    fig, ax = plt.subplots()

    with exp:
        return_val = plot_plume(df, ax=ax, **plot_kwargs)

    assert return_val == ax

    out_file = tmp_path / "fig.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Check this works via the accessor too
    fig, ax = plt.subplots()
    with exp:
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
    exp: contextlib.AbstractContextManager = does_not_raise(),
) -> None:
    fig, ax = plt.subplots()

    with exp:
        return_val = plot_plume_after_calculating_quantiles(df, ax=ax, **method_kwargs)

    assert return_val == ax

    out_file = tmp_path / "fig.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Check this works via the accessor too
    fig, ax = plt.subplots()
    with exp:
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
        rng=np.random.default_rng(seed=11241),
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


def test_plot_plume_extra_palette(
    tmp_path,
    image_regression,
    setup_pandas_accessor,
):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "W")),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=6543),
    )

    method_kwargs = dict(
        quantile_over="run",
        quantiles_plumes=((0.5, 0.5), ((0.05, 0.95), 0.2)),
        hue_var="scenario",
        palette={
            "scenario_0": "tab:green",
            "scenario_1": "tab:purple",
            "scenario_2": "tab:red",
            # Not df
            "scenario_3": "tab:orange",
        },
        style_var="variable",
    )

    check_plots_incl_quantile_calculation(
        df=df,
        method_kwargs=method_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


def test_plot_plume_missing_from_palette(
    tmp_path,
    image_regression,
    setup_pandas_accessor,
):
    df = create_test_df(
        variables=(("variable_1", "K"),),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=85918),
    )

    palette = {
        "scenario_0": "tab:orange",
        # "scenario_1": "tab:blue",
        "scenario_2": "tab:blue",
    }
    method_kwargs = dict(
        quantile_over="run",
        quantiles_plumes=((0.5, 0.5), ((0.05, 0.95), 0.2)),
        hue_var="scenario",
        palette=palette,
    )

    check_plots_incl_quantile_calculation(
        df=df,
        method_kwargs=method_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
        exp=pytest.warns(
            match=re.escape(
                "Some hue values are not in the user-supplied palette, "
                "they will be filled from the default colour cycler instead. "
                f"missing_from_user_supplied={['scenario_1']} "
                f"palette_user_supplied={palette}"
            )
        ),
    )


@pytest.mark.parametrize(
    "quantiles, quantiles_plumes, exp",
    (
        pytest.param(
            [0.05, 0.5, 0.95],
            ((0.45, 0.5), ((0.05, 0.95), 0.2)),
            pytest.warns(match=re.escape("Missing quantiles=(0.45,)")),
            id="missing-line-quantile",
        ),
    ),
)
def test_plot_plume_missing_quantiles(  # noqa: PLR0913
    quantiles, quantiles_plumes, exp, setup_pandas_accessor, image_regression, tmp_path
):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=2,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=112345),
    )

    check_plots(
        df=df.openscm.groupby_except("run")
        .quantile(quantiles)
        .openscm.fix_index_name_after_groupby_quantile(),
        plot_kwargs=dict(quantiles_plumes=quantiles_plumes),
        image_regression=image_regression,
        tmp_path=tmp_path,
        exp=exp,
    )


# def test_plot_plume_units():
#     # Plot two different sets of data with different units
#     # Make sure that the unit handling passes through
#     assert False


def test_plot_plume_option_passing(setup_pandas_accessor, image_regression, tmp_path):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(2025.0, 2150.0),
        rng=np.random.default_rng(seed=85910),
    )

    pdf = (
        df.openscm.groupby_except("run")
        .quantile([0.1685321, 0.5, 0.8355321])
        .openscm.fix_index_name_after_groupby_quantile(new_name="percentile")
        .reset_index(["unit", "percentile"])
    )
    pdf["percentile"] *= 100.0
    pdf = pdf.rename({"unit": "units"}, axis="columns")
    pdf = pdf.set_index(["units", "percentile"], append=True)
    pdf.columns = pdf.columns.astype(float)

    def create_legend(ax, handles) -> None:
        ax.legend(handles=handles, loc="best", handlelength=4)

    plot_kwargs = dict(
        quantiles_plumes=((50.0, 1.0), ((16.85321, 83.55321), 0.3)),
        quantile_var="percentile",
        quantile_var_label="Percent",
        quantile_legend_round=3,
        hue_var="variable",
        hue_var_label="Var",
        palette={
            # Drop out to trigger warning below
            # "variable_1": "tab:green",
            "variable_2": "tab:purple",
        },
        warn_on_palette_value_missing=False,
        style_var="scenario",
        style_var_label="Scen",
        dashes={
            "scenario_0": "--",
            # Drop out to trigger warning below
            # "scenario_1": "-",
            "scenario_2": (0, (5, 3, 5, 1)),
        },
        warn_on_dashes_value_missing=False,
        linewidth=1.5,
        unit_col="units",
        x_label="Year",
        y_label="Value",
        # warn_infer_y_label_with_multi_unit tested elsewhere
        create_legend=create_legend,
        observed=False,
    )

    check_plots(
        df=pdf,
        plot_kwargs=plot_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


def test_plot_plume_after_calculating_quantiles_option_passing(
    setup_pandas_accessor, image_regression, tmp_path
):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(2025.0, 2150.0),
        rng=np.random.default_rng(seed=17583),
    )

    pdf = df.reset_index("unit")
    pdf = pdf.rename({"unit": "units"}, axis="columns")
    pdf = pdf.set_index("units", append=True)
    pdf.columns = pdf.columns.astype(float)

    def create_legend(ax, handles) -> None:
        ax.legend(handles=handles, loc="best", handlelength=4)

    method_kwargs = dict(
        quantile_over="run",
        quantiles_plumes=((0.5, 1.0), ((1.0 / 6.0, 5.0 / 6.0), 0.3)),
        quantile_var_label="Percent",
        quantile_legend_round=3,
        hue_var="variable",
        hue_var_label="Var",
        palette={
            # Drop out to trigger warning below
            # "variable_1": "tab:green",
            "variable_2": "tab:purple",
        },
        warn_on_palette_value_missing=False,
        style_var="scenario",
        style_var_label="Scen",
        dashes={
            "scenario_0": "--",
            # Drop out to trigger warning below
            # "scenario_1": "-",
            "scenario_2": (0, (5, 3, 5, 1)),
        },
        warn_on_dashes_value_missing=False,
        linewidth=1.5,
        unit_col="units",
        x_label="Year",
        y_label="Value",
        # warn_infer_y_label_with_multi_unit tested elsewhere
        create_legend=create_legend,
        observed=False,
    )

    check_plots_incl_quantile_calculation(
        df=pdf,
        method_kwargs=method_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


def test_get_default_colour_cycler_no_matplotlib():
    with patch.dict(sys.modules, {"matplotlib": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=re.escape(
                "`get_default_colour_cycler` requires matplotlib to be installed"
            ),
        ):
            get_default_colour_cycler()
