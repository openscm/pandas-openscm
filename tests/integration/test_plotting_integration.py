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
    PlumePlotter,
    SingleLinePlotter,
    extract_single_unit,
    get_default_colour_cycler,
    get_quantiles,
    get_values_line,
    get_values_plume,
    plot_plume_after_calculating_quantiles_func,
    plot_plume_func,
)
from pandas_openscm.testing import create_test_df

matplotlib = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
pytest_regressions = pytest.importorskip("pytest_regressions")

try:
    import openscm_units
except ImportError:
    openscm_units = None


def check_plots(
    plot_kwargs: dict[str, Any],
    df: pd.DataFrame,
    image_regression: pytest_regressions.image_regression.ImageRegressionFixture,
    tmp_path: Path,
    exp: contextlib.AbstractContextManager = does_not_raise(),
) -> None:
    fig, ax = plt.subplots()

    with exp:
        return_val = plot_plume_func(df, ax=ax, **plot_kwargs)

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

    plt.close()


def check_plots_incl_quantile_calculation(
    method_kwargs: dict[str, Any],
    df: pd.DataFrame,
    image_regression: pytest_regressions.image_regression.ImageRegressionFixture,
    tmp_path: Path,
    exp: contextlib.AbstractContextManager = does_not_raise(),
) -> None:
    fig, ax = plt.subplots()

    with exp:
        return_val = plot_plume_after_calculating_quantiles_func(
            df, ax=ax, **method_kwargs
        )

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

    plt.close()


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
        plot_kwargs=dict(quantiles_plumes=((0.5, 0.8), ((0.05, 0.95), 0.3))),
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


def test_default_ax_auto_creation(tmp_path, image_regression, setup_pandas_accessor):
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=83747),
    )

    res = (
        df.openscm.groupby_except("run")
        .quantile([0.05, 0.5, 0.95])
        .openscm.fix_index_name_after_groupby_quantile()
        .openscm.plot_plume(quantiles_plumes=((0.5, 0.8), ((0.05, 0.95), 0.3)))
    )
    assert isinstance(res, matplotlib.axes.Axes)

    out_file = tmp_path / "fig.png"
    plt.savefig(out_file, bbox_extra_artists=(res.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


def test_plot_plume_no_labels(tmp_path, image_regression, setup_pandas_accessor):
    df = create_test_df(
        variables=(("variable_1", "K"),),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=84747),
    )

    check_plots(
        df=df.openscm.groupby_except("run")
        .quantile([0.05, 0.5, 0.95])
        .openscm.fix_index_name_after_groupby_quantile(),
        plot_kwargs=dict(
            y_label=None,
            x_label=None,
            quantiles_plumes=((0.5, 0.8), ((0.05, 0.95), 0.3)),
        ),
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


def test_plot_plume_with_other_plot_calls(
    tmp_path, image_regression, setup_pandas_accessor
):
    fig, ax = plt.subplots()

    before_handles = ax.plot(
        np.arange(1955.0, 1975.0, 2.5),
        8 - np.arange(8),
        color="gray",
        label="before",
        linewidth=2.5,
        linestyle="--",
        zorder=3.1,
    )

    df = (
        create_test_df(
            variables=(("variable_1", "K"), ("variable_2", "K")),
            n_scenarios=3,
            n_runs=10,
            timepoints=np.arange(1950.0, 1965.0),
            rng=np.random.default_rng(seed=999),
        )
        .openscm.groupby_except("run")
        .quantile([0.05, 0.5, 0.95])
        .openscm.fix_index_name_after_groupby_quantile()
    )

    df.openscm.plot_plume(
        ax=ax,
        quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.2)),
    )

    x_after = np.linspace(1940.0, 1970.0, 101)
    after_handles = ax.plot(
        x_after,
        np.sin(x_after) + 5.0 + np.linspace(0, 30.0, x_after.size),
        color="pink",
        label="after",
        linewidth=3,
        zorder=3.1,
    )

    # This is mucking around.
    # If you wanted to do something like this,
    # much simpler to just get the PlumePlotter object
    # and do the handles that way.
    # This at least shows that this sort of thing is possible.
    new_handles = [
        *before_handles,
        *after_handles,
        *ax.get_legend().legend_handles,
    ]
    ax.legend(handles=new_handles, loc="lower left", bbox_to_anchor=(1.05, 0.25))

    out_file = tmp_path / "fig.png"
    plt.savefig(
        out_file,
        bbox_extra_artists=(ax.get_legend(),),
        bbox_inches="tight",
    )

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    plt.close()


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
            dict(unit_var=None),
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
        warn_infer_y_label_with_multi_unit=False,
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


def test_plot_plume_extra_dashes(
    tmp_path,
    image_regression,
    setup_pandas_accessor,
):
    df = create_test_df(
        variables=(("variable_1", "W"), ("variable_2", "W")),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=6543),
    )

    method_kwargs = dict(
        quantile_over="run",
        quantiles_plumes=((0.5, 0.5), ((0.05, 0.95), 0.2)),
        hue_var="scenario",
        style_var="variable",
        dashes={
            # Not in df
            "variable_0": "-",
            "variable_1": "--",
            "variable_2": ":",
        },
    )

    check_plots_incl_quantile_calculation(
        df=df,
        method_kwargs=method_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
    )


def test_plot_plume_missing_from_dashes(
    tmp_path,
    image_regression,
    setup_pandas_accessor,
):
    df = create_test_df(
        variables=(("variable_1", "W"), ("variable_2", "W")),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=85919),
    )

    dashes = {
        # "variable_1": "--",
        "variable_2": ":",
    }
    method_kwargs = dict(
        quantile_over="run",
        quantiles_plumes=((0.5, 0.5), ((0.05, 0.95), 0.2)),
        hue_var="scenario",
        style_var="variable",
        dashes=dashes,
    )

    check_plots_incl_quantile_calculation(
        df=df,
        method_kwargs=method_kwargs,
        image_regression=image_regression,
        tmp_path=tmp_path,
        exp=pytest.warns(
            match=re.escape(
                "Some style values are not in the user-supplied dashes, "
                "they will be filled from the default dash cycler instead. "
                f"missing_from_user_supplied={['variable_1']} "
                f"dashes_user_supplied={dashes}"
            )
        ),
    )


@pytest.mark.parametrize(
    "quantiles, quantiles_plumes, exp",
    (
        pytest.param(
            [0.05, 0.5, 0.95],
            ((0.45, 0.5), ((0.05, 0.95), 0.2)),
            pytest.warns(match=r"Quantiles missing.*missing_quantiles=\[0.45\]"),
            id="missing-line-quantile",
        ),
        pytest.param(
            [0.05, 0.5, 0.95],
            ((0.5, 0.5), ((0.15, 0.95), 0.2)),
            pytest.warns(match=r"Quantiles missing.*missing_quantiles=\[0.15\]"),
            id="missing-plume-quantile",
        ),
        pytest.param(
            [0.05, 0.5, 0.95],
            ((0.5, 0.5), ((0.15, 0.85), 0.2)),
            pytest.warns(match=r"Quantiles missing.*missing_quantiles=\[0.15, 0.85\]"),
            id="missing-plumes-quantile",
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


def test_plot_plume_missing_multiple_quantiles(
    setup_pandas_accessor,
    image_regression,
    tmp_path,
    recwarn,
):
    quantiles = [0.25, 0.75]
    quantiles_plumes = ((0.5, 0.5), ((0.15, 0.85), 0.2), ((0.25, 0.75), 0.4))

    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=2,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        rng=np.random.default_rng(seed=118844),
    )

    check_plots(
        df=df.openscm.groupby_except("run")
        .quantile(quantiles)
        .openscm.fix_index_name_after_groupby_quantile(),
        plot_kwargs=dict(quantiles_plumes=quantiles_plumes),
        image_regression=image_regression,
        tmp_path=tmp_path,
    )

    assert len(recwarn) == 16
    for w in recwarn:
        assert any(
            m in str(w.message)
            for m in (
                "missing_quantiles=[0.5]",
                "missing_quantiles=[0.15, 0.85]",
            )
        )


def test_plot_plume_option_passing(setup_pandas_accessor, image_regression, tmp_path):
    openscm_units = pytest.importorskip("openscm_units")
    openscm_units.unit_registry.setup_matplotlib(enable=True)

    df = create_test_df(
        variables=(("variable_1", "tCO2"), ("variable_2", "dtCO2")),
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
        unit_var="units",
        unit_aware=openscm_units.unit_registry,
        time_units="month",
        x_label=None,  # let unit-awareness take over
        y_label=None,  # let unit-awareness take over
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

    # Teardown
    openscm_units.unit_registry.setup_matplotlib(enable=False)


def test_plot_plume_after_calculating_quantiles_option_passing(
    setup_pandas_accessor, image_regression, tmp_path
):
    openscm_units = pytest.importorskip("openscm_units")
    openscm_units.unit_registry.setup_matplotlib(enable=True)

    df = create_test_df(
        variables=(("variable_1", "ctCO2"), ("variable_2", "dtCO2")),
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
        unit_var="units",
        unit_aware=openscm_units.unit_registry,
        time_units="decade",
        x_label=None,  # let unit-awareness take over
        y_label=None,  # let unit-awareness take over
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

    # Teardown
    openscm_units.unit_registry.setup_matplotlib(enable=False)


@pytest.mark.parametrize(
    "unit_aware, variables",
    (
        pytest.param(
            True,
            (("variable_1", "cW"), ("variable_2", "mW")),
            id="default-unit-registry",
        ),
        pytest.param(
            getattr(openscm_units, "unit_registry", None),
            (("variable_1", "GtC"), ("variable_2", "GtCO2")),
            marks=pytest.mark.skipif(
                openscm_units is None, reason="openscm_units not installed"
            ),
            id="user-provided-unit-registry",
        ),
    ),
)
def test_plot_plume_unit_aware(
    unit_aware, variables, setup_pandas_accessor, image_regression, tmp_path
):
    """
    Make sure that we can do unit-aware plots

    In other words, even if the units are different,
    if they're compatible, they're plotted with the same units.
    """
    if isinstance(unit_aware, bool):
        pint = pytest.importorskip("pint")
        ur = pint.get_application_registry()
    else:
        ur = unit_aware

    ur.setup_matplotlib(enable=True)

    fig, ax = plt.subplots()

    res = (
        create_test_df(
            variables=variables,
            n_scenarios=3,
            n_runs=10,
            timepoints=np.arange(1950.0, 2050.0),
            rng=np.random.default_rng(seed=8888),
        )
        .openscm.groupby_except("run")
        .quantile([0.05, 0.5, 0.95])
        .openscm.fix_index_name_after_groupby_quantile()
        .openscm.plot_plume(
            ax=ax,
            quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.2)),
            unit_aware=unit_aware,
            time_units="yr",
        )
    )

    assert res == ax

    out_file = tmp_path / "fig.png"
    plt.savefig(
        out_file,
        bbox_extra_artists=(ax.get_legend(),),
        bbox_inches="tight",
    )

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    plt.close()

    # Teardown
    ur.setup_matplotlib(enable=False)


def test_plot_plume_unit_aware_incompatible_units(setup_pandas_accessor):
    """
    Make sure that we can do unit-aware plots and errors are caught

    In other words, if the units are incompatible,
    then trying to plot on the same axes raises an error.
    """
    pint = pytest.importorskip("pint")

    ur = pint.get_application_registry()
    ur.setup_matplotlib(enable=True)

    df_m = create_test_df(
        variables=(("variable_1", "m"),),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(1950.0, 2050.0),
        rng=np.random.default_rng(seed=8888),
    )

    df_kg = create_test_df(
        variables=(("variable_2", "kg"),),
        n_scenarios=3,
        n_runs=10,
        timepoints=np.arange(1950.0, 2050.0),
        rng=np.random.default_rng(seed=8889),
    )

    _, ax = plt.subplots()

    # Plot first df
    df_m_grouped = (
        df_m.openscm.groupby_except("run")
        .quantile([0.05, 0.5, 0.95])
        .openscm.fix_index_name_after_groupby_quantile()
    )
    df_m_grouped.openscm.plot_plume(
        ax=ax,
        quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.2)),
        unit_aware=True,
        time_units="yr",
    )

    with pytest.raises(
        matplotlib.units.ConversionError,
        match=re.escape("Failed to convert value(s) to axis units"),
    ):
        # Plotting an incompatible unit raises.
        # Test plot_plume accesor (to double check argument passing)
        (
            df_kg.openscm.groupby_except("run")
            .quantile([0.05, 0.5, 0.95])
            .openscm.fix_index_name_after_groupby_quantile()
        ).openscm.plot_plume(
            ax=ax,
            quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.2)),
            unit_aware=True,
            time_units="yr",
        )

    with pytest.raises(
        matplotlib.units.ConversionError,
        match=re.escape("Failed to convert value(s) to axis units"),
    ):
        # Test that different x-axis also fails
        df_m_grouped.openscm.plot_plume(
            ax=ax,
            quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.2)),
            unit_aware=True,
            time_units="kg",
        )

    with pytest.raises(
        matplotlib.units.ConversionError,
        match=re.escape("Failed to convert value(s) to axis units"),
    ):
        # Test plot_plume_after_calculating_quantiles accesor
        # (to double check argument passing)
        df_kg.openscm.plot_plume_after_calculating_quantiles(
            ax=ax,
            quantile_over="run",
            quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.2)),
            unit_aware=True,
            time_units="yr",
        )

    # Teardown
    ur.setup_matplotlib(enable=False)


def test_get_default_colour_cycler_no_matplotlib():
    with patch.dict(sys.modules, {"matplotlib": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=re.escape(
                "`get_default_colour_cycler` requires matplotlib to be installed"
            ),
        ):
            get_default_colour_cycler()


def test_extract_single_unit_raises_if_more_than_one():
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "W")),
        n_scenarios=1,
        n_runs=1,
        timepoints=np.arange(1950.0, 1965.0),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("['K', 'W']"),
    ):
        extract_single_unit(df, unit_var="unit")


def test_get_values_line_unit_aware_no_pint():
    df = create_test_df(
        variables=(("variable_1", "K"), ("variable_2", "K")),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
    )

    with patch.dict(sys.modules, {"pint": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=re.escape(
                "`get_values_line(..., unit_aware=True, ...)` "
                "requires pint to be installed"
            ),
        ):
            get_values_line(df, unit_aware=True, unit_var="unit", time_units="yr")


def test_get_values_plume_unit_aware_no_pint(setup_pandas_accessor):
    df = (
        create_test_df(
            variables=(("variable_1", "K"), ("variable_2", "K")),
            n_scenarios=5,
            n_runs=10,
            timepoints=np.arange(1950.0, 1965.0),
        )
        .openscm.groupby_except("run")
        .quantile([0.25, 0.75])
        .openscm.fix_index_name_after_groupby_quantile()
    )

    with patch.dict(sys.modules, {"pint": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=re.escape(
                "`get_values_plume(..., unit_aware=True, ...)` "
                "requires pint to be installed"
            ),
        ):
            get_values_plume(
                df,
                quantiles=[0.25, 0.75],
                quantile_var="quantile",
                unit_aware=True,
                unit_var="unit",
                time_units="yr",
            )


def test_generate_legend_handles_no_matplotlib():
    pp = PlumePlotter(
        lines=[],
        plumes=[],
        hue_var_label="a",
        style_var_label="b",
        quantile_var_label="c",
        palette={},
        dashes=None,
        x_label=None,
        y_label=None,
    )
    with patch.dict(sys.modules, {"matplotlib": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=re.escape(
                "`generate_legend_handles` requires matplotlib to be installed"
            ),
        ):
            pp.generate_legend_handles()


def test_plot_no_matplotlib():
    pp = PlumePlotter(
        lines=[],
        plumes=[],
        hue_var_label="a",
        style_var_label="b",
        quantile_var_label="c",
        palette={},
        dashes=None,
        x_label=None,
        y_label=None,
    )
    with patch.dict(sys.modules, {"matplotlib": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=re.escape("`plot(ax=None, ...)` requires matplotlib to be installed"),
        ):
            pp.plot()


def test_single_line_plotter_wrong_shape_y_vals():
    error_msg = re.escape(
        "`y_vals` must have the same shape as `x_vals`. "
        "Received `y_vals` with shape (2,) while `x_vals` has shape (3,)"
    )
    with pytest.raises(AssertionError, match=error_msg):
        SingleLinePlotter(
            np.array([1, 2, 3]),
            np.array([1, 2]),
            quantile=0.3,
            linewidth=1.0,
            linestyle="-",
            color="blue",
            alpha=0.3,
        )
