# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to make a plume plot
#
# In this notebook we describe how to make a plume plot using pandas-OpenSCM.
# If you don't know what a plume plot is, read on, it will be shown soon.

# %% [markdown]
# ## Imports

# %%
import traceback

import matplotlib.pyplot as plt
import matplotlib.units
import numpy as np
import openscm_units

from pandas_openscm import register_pandas_accessors
from pandas_openscm.plotting import PlumePlotter
from pandas_openscm.testing import create_test_df

# %% [markdown]
# ## Setup

# %%
# Register the openscm accessor for pandas objects
# (we don't do this on import
# as we have had bad experiences with implicit behaviour like that)
register_pandas_accessors()

# %% [markdown]
# ## Basics
#
# Imagine we start with some data that has multiple realisations.
# For example, multiple runs from a simple climate model.

# %%
df_basic = create_test_df(
    variables=(("Warming", "K"),),
    n_scenarios=5,
    n_runs=10,
    timepoints=np.arange(1950.0, 1965.0),
)
df_basic

# %% [markdown]
# From this data, we can calculate statistics over the runs.
# For example, different quantiles.

# %%
df_basic_quantiles = (
    df_basic.openscm.groupby_except("run")
    .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    .openscm.fix_index_name_after_groupby_quantile()
)
df_basic_quantiles

# %% [markdown]
# Then we can make a plume plot,
# i.e. a plot which shows ranges between quantiles
# and/or individual quantiles.

# %%
df_basic_quantiles.openscm.plot_plume(
    quantiles_plumes=(
        # The syntax here is:
        #     (quantile(s), alpha)
        # If quantile(s) is a float, we plot a line.
        # If quantile(s) is a tuple, we plot a plume between the given quantiles.
        (0.5, 0.8),
        ((0.25, 0.75), 0.5),
        ((0.05, 0.95), 0.3),
    ),
)

# %% [markdown]
# With
# [plot_plume_after_calculating_quantiles](../../api/pandas_openscm/plotting/#pandas_openscm.plotting.plot_plume_after_calculating_quantiles),
# you don't have to calculate the quantiles first yourself.

# %%
df_basic.openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run",
    quantiles_plumes=(
        # Plot two quantiles as lines, rather than just one
        (0.25, 0.4),
        (0.75, 0.8),
        ((0.1, 0.9), 0.3),
    ),
)

# %% [markdown]
# By default, we try and set the x- and y-labels.
# It is possible to provide your own values.

# %%
df_basic_quantiles.openscm.plot_plume(
    quantiles_plumes=(((0.05, 0.95), 0.3),),
    x_label="Year",
    y_label="Warming",
)

# %% [markdown]
# Beyond splitting the data into different colours,
# it is also possible to split along another dimension/facet too,
# using the style variable.

# %%
df_multi_facet = create_test_df(
    variables=(("Warming land", "K"), ("Warming ocean", "K")),
    n_scenarios=5,
    n_runs=10,
    timepoints=np.arange(1850.0, 2024.0),
)

df_multi_facet.openscm.plot_plume_after_calculating_quantiles(
    style_var="variable",
    quantile_over="run",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.1, 0.9), 0.3),
    ),
)

# %% [markdown]
# ## Summary so far
#
# The functionality shown up to here is the key functionality.
# If all you need to do is basic plots, this is all you need.
# If you are looking to do fancier customisation of plots,
# read on.

# %% [markdown]
# ## Advanced topics

# %% [markdown]
# ### Fine-grained control
#
# Almost every aspect of the plot can be altered as you wish.
# Below we show the full range of options that can be passed.
# These can be passed via the accessors we have used in these docs
# or directly via the underlying functions.

# %% [markdown]
# Firstly, we set up some specific data.
# For example, data without quantiles but rather very specific percentiles
# (e.g. someone sent you pre-processed data).

# %%
df_advanced = create_test_df(
    variables=(("variable_1", "tCO2"), ("variable_2", "dtCO2")),
    n_scenarios=3,
    n_runs=10,
    timepoints=np.arange(2025.0, 2150.0),
)

# Imagine that we don't have quantiles, rather very specific percentiles
# (e.g. someone sent you pre-processed data).
plot_df = (
    df_advanced.openscm.groupby_except("run")
    .quantile([0.1685321, 0.5, 0.8355321])
    .openscm.fix_index_name_after_groupby_quantile(new_name="percentile")
    .reset_index(["unit", "percentile"])
)
plot_df["percentile"] *= 100.0
plot_df = plot_df.rename({"unit": "units"}, axis="columns")
plot_df = plot_df.set_index(["units", "percentile"], append=True)
plot_df.columns = plot_df.columns.astype(float)
plot_df


# %%
# You can control how the legend is created by passing
# in your own function
# (this allows you to also disable legend creation
# by passing in a no-op).
def create_legend(ax, handles) -> None:
    """Create custom legend"""
    ax.legend(handles=handles, loc="best", handlelength=4)


_, axes = plt.subplots(ncols=2)

plot_df.openscm.plot_plume(
    # Specify the axes to plot on
    ax=axes[1],
    quantiles_plumes=((50.0, 1.0), ((16.85321, 83.55321), 0.3)),
    # Specify that the quantiles are in a column that is not the default
    quantile_var="percentile",
    # Custom label for the quantiles in the legend
    quantile_var_label="Percent",
    # Rounding when displaying the quantiles in the legend
    quantile_legend_round=3,
    # Variable that defines colour groupings
    hue_var="variable",
    # Heading to use for the colour groupings in the legend
    hue_var_label="Var",
    # Palette to use.
    palette={
        # Drop out to trigger warning below
        # "variable_1": "tab:green",
        "variable_2": "tab:cyan",
    },
    # Notice that we don't supply all values,
    # which is why we get a warning.
    # This can be disabled with the line below.
    # warn_on_palette_value_missing=False,
    # Variable that defines style groupings
    style_var="scenario",
    # Heading to use for the style groupings in the legend
    style_var_label="Scen",
    # Dashes to use for different values of the style variable.
    dashes={
        "scenario_0": "--",
        # Drop out to trigger warning below
        # "scenario_1": "-",
        "scenario_2": (0, (5, 3, 5, 1)),
    },
    # Notice that we don't supply all values,
    # which is why we get a warning.
    # This can be disabled with the line below.
    # warn_on_dashes_value_missing=False,
    # Linewidth to use for lines
    linewidth=1.5,
    # The variable (column in the data's multi-index)
    # that holds unit information.
    unit_var="units",
    # In the output, notice that a warning is raised,
    # which says that the y-label is not automatically set
    # because there is more than unit in the data.
    # This warning can be silenced with the line below.
    # (More on unit support below).
    # warn_infer_y_label_with_multi_unit=False,
    # Inject our own legend creation
    create_legend=create_legend,
)

# %% [markdown]
# ### Unit-aware plotting
#
# Matplotlib and pint support plotting with units
# ([stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html),
# [last version that we checked at the time of writing](https://pint.readthedocs.io/en/0.24.4/user/plotting.html)).

# %% [markdown]
# Firstly, set-up matplotlib to use the unit registry.

# %%
openscm_units.unit_registry.setup_matplotlib(enable=True)

# %% [markdown]
# Now we can tell our plottting functions to do unit-aware plotting.
# If we plot now, the units will appear on the output axis automatically
# and data will be plotted in the same units.

# %%
df_compatible_units = create_test_df(
    # Notice the different, but compatible, units here
    variables=(("Emissions (ct)", "ctCO2"), ("Emissions (dt)", "dtCO2")),
    n_scenarios=5,
    n_runs=10,
    timepoints=np.arange(1850.0, 2024.0),
)

df_compatible_units.openscm.plot_plume_after_calculating_quantiles(
    style_var="variable",
    quantile_over="run",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.1, 0.9), 0.3),
    ),
    # Opt into unit-aware plotting
    unit_aware=openscm_units.unit_registry,
    # Also specify the time axis
    time_units="yr",
)

# %% [markdown]
# Without unit-aware plotting, we don't see the true difference in the data.

# %%
df_compatible_units.openscm.plot_plume_after_calculating_quantiles(
    style_var="variable",
    quantile_over="run",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.1, 0.9), 0.3),
    ),
    # # Opt into unit-aware plotting
    # unit_aware=openscm_units.unit_registry,
    # # Also specify the time axis
    # time_units="yr",
    # We know we have different units,
    # we don't need the warning here
    warn_infer_y_label_with_multi_unit=False,
)

# %% [markdown]
# Unit-aware plotting is extremely helpful for avoiding plotting incompatible units.

# %%
ax = df_compatible_units.openscm.plot_plume_after_calculating_quantiles(
    style_var="variable",
    quantile_over="run",
    quantiles_plumes=((0.5, 0.8),),
    unit_aware=openscm_units.unit_registry,
    time_units="yr",
)

# If we now try and plot data with other units on these axes,
# an error is raised.
try:
    df_multi_facet.openscm.plot_plume_after_calculating_quantiles(
        ax=ax,
        style_var="variable",
        quantile_over="run",
        quantiles_plumes=(((0.1, 0.9), 0.3),),
        unit_aware=openscm_units.unit_registry,
        time_units="yr",
    )
except matplotlib.units.ConversionError:
    traceback.print_exc(limit=0)
    # Note that this data doesn't appear on the output plot

# %% [markdown]
# ### PlumePlotter
#
# Underneath these plots is the
# [PlumePlotter](../../api/pandas_openscm/plotting/#pandas_openscm.plotting.PlumePlotter)
# class.
# If you want really fine-grained control of your plots,
# we recommend dropping down to using this class directly.

# %%
# Initialise our plume_plotter
plume_plotter = PlumePlotter.from_df(
    df_multi_facet.openscm.groupby_except("run")
    .quantile([0.1, 0.5, 0.9])
    .openscm.fix_index_name_after_groupby_quantile(),
    style_var="variable",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.1, 0.9), 0.3),
    ),
    unit_aware=openscm_units.unit_registry,
    time_units="yr",
    # Don't infer any labels for the axes
    x_label=None,
    y_label=None,
)
print(f"{len(plume_plotter.lines)=}")
print(f"{len(plume_plotter.plumes)=}")

# %%
df_multi_facet

# %%
# Now make our plots
fig, ax = plt.subplots()
ax.xaxis.set_units(openscm_units.unit_registry("yr").u)
ax.yaxis.set_units(openscm_units.unit_registry("K").u)

# Some other data we want to plot
Q = openscm_units.unit_registry.Quantity
history_lines = ax.plot(
    Q(np.arange(0, 30 * 12.0) + 1998.0 * 12, "month"),
    Q(
        10_000 * np.sin(np.arange(0, 30 * 12.0) / 20.0) + 100 * np.arange(0, 30 * 12.0),
        "mK",
    ),
    label="history",
    color="black",
    zorder=3,
    linewidth=3,
)
scatter_points = ax.scatter(
    Q([1910, 1930, 1980], "yr"),
    Q([8, 3, 15], "K"),
    label="interesting points",
    color="tab:pink",
    zorder=3,
    linewidth=3,
    marker="x",
    s=120,
)

# Plot using the plume plotter, without making a legend
plume_plotter.plot(ax=ax, create_legend=lambda x, y: ...)
# Get the legend items from the plume plotter
plume_legend_items = plume_plotter.generate_legend_handles(quantile_legend_round=3)

# Plot something else
other_history_lines = ax.plot(
    Q(np.arange(0, 30 * 12.0) + 1998.0 * 12, "month"),
    Q(
        10_000 * (1 - np.sin(np.arange(0, 30 * 12.0) / 20.0))
        + 100 * np.arange(0, 30 * 12.0),
        "mK",
    ),
    label="history data 2",
    color="gray",
    zorder=3,
    linewidth=3,
)

# Now make a legend
ax.legend(
    handles=[
        *history_lines,
        *other_history_lines,
        scatter_points,
        *plume_legend_items,
    ],
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
)
# Can also modify all other parts of the plot (as normal)
ax.set_xlabel("Year")
ax.set_ylabel("K")
ax.set_title("Demo")

# %% [markdown]
# ### Summary
#
# We use the plume plotting a lot
# Like the rest of the library, it is in a work progress.
# If there is a feature you would like or a bug or anything else,
# please [raise an issue](https://github.com/openscm/pandas-openscm/issues).
