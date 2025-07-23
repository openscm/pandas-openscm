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
# # Unit conversion
#
# Here we detail pandas-openscm's unit conversion support.

# %% [markdown]
# ## Imports

# %%
import traceback

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pint

from pandas_openscm import register_pandas_accessor
from pandas_openscm.testing import create_test_df
from pandas_openscm.unit_conversion import (
    AmbiguousTargetUnitError,
    convert_unit_from_target_series,
)

# %% [markdown]
# ## Setup

# %%
# Register the openscm accessor for pandas objects
# (we don't do this on import
# as we have had bad experiences with implicit behaviour like that)
register_pandas_accessor()

# %% [markdown]
# ## Basics

# %% [markdown]
# ### Convert all data to a given unit

# %% [markdown]
# Imagine we start with some data.

# %%
df_basic = create_test_df(
    variables=(("Warming", "K"),),
    n_scenarios=2,
    n_runs=2,
    timepoints=np.arange(1950.0, 1965.0),
)
df_basic

# %% [markdown]
# If we want to convert the entire dataset to a different unit,
# we can simply call `convert_unit` with the desired unit.

# %%
df_basic.openscm.convert_unit("degC")

# %% [markdown]
# By default, this assumes that the unit information
# is in an index level called 'unit'.
# If this isn't the case, the `unit_level` argument should be used.

# %%
df_other_unit_col = create_test_df(
    variables=(("Warming", "K"),),
    n_scenarios=2,
    n_runs=2,
    timepoints=np.arange(1950.0, 1965.0),
).rename_axis(["scenario", "variable", "run", "units"])

# Notice that unit information is in a column called "units" not "unit"
df_other_unit_col

# %%
df_other_unit_col.openscm.convert_unit("kK", unit_level="units")

# %% [markdown]
# ### More specific conversions
#
# Above we have shown how to convert the entire dataset to a given unit.
# This works well when such a conversion is possible.
# However, we often have cases where different timeseries have different dimensionality,
# therefore cannot all be converted to the same unit.

# %% [markdown]
# Once again, start with some example data.
# Here we have data in different units.

# %%
df_multi_unit = create_test_df(
    variables=(
        ("Warming", "K"),
        ("Ocean Heat Content", "ZJ"),
        ("SLR", "mm"),
    ),
    n_scenarios=2,
    n_runs=2,
    timepoints=np.arange(1950.0, 1965.0),
)
df_multi_unit

# %% [markdown]
# If we try to convert everything to a single unit,
# we will get a dimensionality error for whatever units aren't compatible.

# %%
try:
    df_multi_unit.openscm.convert_unit("cm")
except pint.DimensionalityError:
    traceback.print_exc(limit=0)

# %% [markdown]
# To support unit conversion in such a case,
# we can do one of three things:
#
# 1. filter the data first before converting the unit
#     - this obviously defeats the purpose of this API a bit,
#       as you would end up filtering, converting and recombining
#       everywhere. As a result, we ignore this option from here.
# 2. specify the conversion as a mapping from the current unit to the desired unit
# 3. specify the conversion as a `pd.Series` of the units we would like to end up with

# %% [markdown]
# #### Specifying the unit as a mapping
#
# One option is to specify the conversion
# as a mapping from the current unit to the desired unit.
# This option is useful if all you need to specify the desired unit
# is the current unit.
# Any current units which don't appear in the mapping
# are simply left alone i.e. the data for these rows is simply returned as is.
# The API is quite straightforward and demonstrated below.

# %%
# Note also that no conversion is done for temperature (units of K)
# as "K" does not appear in the mapping
df_multi_unit.openscm.convert_unit({"mm": "cm", "ZJ": "PJ"})

# %% [markdown]
# The main thing to be careful of here is
# that you don't have a typo in your current unit (i.e. mapping key).
# If you do have a typo then,
# *silently*, no conversion will be done
# which may cause you confusion in later code
# (if you expected the conversion to be done and it turns out it hadn't been).

# %%
# There is a typo, "zJ" is given below rather than "ZJ"
# so the ocean heat content data is not converted.
# This happens silently i.e. no warning or error.
df_multi_unit.openscm.convert_unit({"mm": "cm", "zJ": "PJ"})

# %% [markdown]
# #### Specifying the unit as a series
#
# If you want really fine-grained, i.e. timeseries-level, control
# then you can also specify the desired units as a `pd.Series`.
# The `pd.Series` should specify the desired unit for each timeseries
# and have an index which matches the data's index
# except for the unit-information level, which should be included.
#
# This is the hardest to set up,
# but gives you the most control in exchange.
# As for the mapping option,
# any timeseries for which the desired unit is not specified
# are simply returned as they are.

# %%
# There are lots of ways to make a series like this.
# Here we go with a hand-woven, but simple option.
# For your own work, you may want/need something
# that includes much more programming and logic.
desired_unit = pd.Series(
    ["mK", "PJ", "cm", "cm"],
    index=pd.MultiIndex.from_tuples(
        [
            ("scenario_0", "Warming", 0),
            ("scenario_0", "Ocean Heat Content", 1),
            ("scenario_0", "SLR", 0),
            ("scenario_1", "SLR", 1),
        ],
        # Not unit level here
        names=["scenario", "variable", "run"],
    ),
)
desired_unit

# %%
# Note that only the rows which appear in `desired_unit`
# are converted, all others are unchanged.
df_multi_unit.openscm.convert_unit(desired_unit)

# %% [markdown]
# As above, the main gotcha is *silently* not doing conversions.
# If you make typos in the specification, this will happen.
# Given that the specification, such typos can be much harder to spot.

# %%
desired_unit_typo = pd.Series(
    ["mK", "PJ", "cm", "cm"],
    index=pd.MultiIndex.from_tuples(
        [
            ("scenario_0", "Warming", 0),
            ("scenario_0", "Ocean Heat Content", 1),
            # Typo here
            ("scenario_0", "SLr", 0),
            ("scenario_1", "SLR", 1),
        ],
        # Not unit level here
        names=["scenario", "variable", "run"],
    ),
)
desired_unit_typo

# %%
# Note that scenario_0, SLR, run 0 isn't converted because of the typo
df_multi_unit.openscm.convert_unit(desired_unit_typo)

# %% [markdown]
# If you are trying to figure out why something isn't being converted,
# pandas provides some quite helpful APIs.

# %%
rows_that_wont_be_used = desired_unit_typo.index.difference(
    df_multi_unit.index.droplevel("unit")
)
rows_that_wont_be_used

# %% [markdown]
# ## Unit registries and pint
#
# The unit conversion is all done with the
# [pint](https://pint.readthedocs.io/) package by default.
# Pint is built around the idea of
# [unit registries](https://pint.readthedocs.io/en/stable/getting/tutorial.html#initializing-a-registry).
# The registry to use can be passed via the `ur` argument.
# If it is not specified, we use whatever is returned from
# `pint.get_application_registry()`.
# This is pint's way of setting the default registry for whatever you are doing.
# By default, it returns pint's default registry
# but you can set a different registry for whatever work you're doing
# with `pint.set_application_registry()`.

# %% [markdown]
# If you're doing climate work, especially related to emissions,
# you often want 'emissions units' like "Mt CO2/yr".
# These are not recognised by default by Pint
# so you get errors if you try to convert them.

# %%
df_emissions = create_test_df(
    variables=(
        ("co2", "Mt CO2 / yr"),
        ("ch4", "Mt CH4 / yr"),
        ("hfc23", "kt HFC23 / yr"),
    ),
    n_scenarios=2,
    n_runs=1,
    timepoints=np.arange(1950.0, 1965.0),
).reset_index("run", drop=True)
df_emissions

# %% [markdown]
# As was written above,
# the default unit registry does not know about emissions units
# so if we try to convert this data,
# we receive an error.

# %%
try:
    df_emissions.openscm.convert_unit({"Mt CO2 / yr": "GtC / yr"})
except pint.UndefinedUnitError:
    traceback.print_exc(limit=0)

# %% [markdown]
# If we specify [openscm-units](https://openscm-units.readthedocs.io/en/latest/)'
# registry instead, the conversion will work.

# %%
df_emissions.openscm.convert_unit(
    {"Mt CO2 / yr": "GtC / yr"}, ur=openscm_units.unit_registry
)

# %% [markdown]
# If we set the application registry to openscm-units' registry,
# then we do not need to pass the registry
# every time we want to do such a conversion.

# %%
pint.set_application_registry(openscm_units.unit_registry)

# %%
# Now the conversion works without specifying the registry
df_emissions.openscm.convert_unit({"Mt CO2 / yr": "GtC / yr"})

# %% [markdown]
# ### Contexts
#
# Pint supports the idea of [contexts](https://pint.readthedocs.io/en/stable/user/contexts.html).
# Within a context, conversions that would normally not be allowed can be allowed.
# Pint's docs give good examples of cases where this is useful.
# For emissions work, the key one is CO<sub>2</sub>-equivalent units.
# Thanks to Pint's contexts and the unit conversion API,
# converting to CO<sub>2</sub>-equivalent units becomes trivial.

# %%
with openscm_units.unit_registry.context("AR6GWP100"):
    df_emissions_co2_eq = df_emissions.openscm.convert_unit(
        "Mt CO2 / yr", ur=openscm_units.unit_registry
    )

df_emissions_co2_eq

# %%
# From here, calculating total CO2-equivalent emissions
# is then trivial, e.g.
df_emissions_co2_eq.openscm.groupby_except("variable").sum().pix.assign(variable="ghg")

# %% [markdown]
# ## Convert unit like
#
# A common scenario is wanting to compare two datasets.
# In such cases, life is much easier if they have the same unit.
# To support this case, we provide the `convert_unit_like` API.
# This is essentially just a wrapper around
# `convert_unit_from_target_series`, that figures out the desired units
# based on the data which we would like to match.
# If the logic included in `convert_unit_like` doesn't fit your use case,
# then we suggest making your desired units by hand and then directly using
# `convert_unit_from_target_series` or `convert_unit` instead.

# %% [markdown]
# Let's imagine we have scenario data like the below.

# %%
df_scenarios = create_test_df(
    variables=(
        ("co2", "Mt CO2 / yr"),
        ("ch4", "Mt CH4 / yr"),
        ("hfc23", "kt HFC23 / yr"),
    ),
    n_scenarios=2,
    n_runs=1,
    timepoints=np.arange(2025.0, 2100.0 + 1.0),
).reset_index("run", drop=True)
df_scenarios

# %% [markdown]
# Then we have some historical data.

# %%
df_history = create_test_df(
    variables=(
        ("co2", "Gt CO2 / yr"),
        ("ch4", "kt CH4 / yr"),
        ("hfc23", "t HFC23 / yr"),
    ),
    n_scenarios=1,
    n_runs=1,
    timepoints=np.arange(1950.0, 2024.0 + 1.0),
).reset_index(["run", "scenario"], drop=True)
df_history

# %% [markdown]
# We can simply convert the scenario data to have the same units of the history
# with `convert_unit_like`.

# %%
df_scenarios_like_history = df_scenarios.openscm.convert_unit_like(df_history)
df_scenarios_like_history

# %% [markdown]
# For scenarios like this, where the units are clear,
# we can also do the reverse operation.

# %%
df_history.openscm.convert_unit_like(df_scenarios)

# %% [markdown]
# However, if the scenarios themselves had different units,
# then the target unit would be ambiguous and we would get an error.

# %%
df_scenarios_differing_units = df_scenarios.openscm.set_index_levels(
    {
        "unit": [
            "Gt CO2 / yr",
            "kt CH4 / yr",
            "kt HFC23 / yr",
            "Mt CO2 / yr",
            "Mt CH4 / yr",
            "kt HFC23 / yr",
        ]
    }
)

# Note that e.g. scenario_0 uses Gt CO2 / yr for co2
# while scenario_1 uses Mt CO2 / yr
df_scenarios_differing_units

# %%
try:
    df_history.openscm.convert_unit_like(df_scenarios_differing_units)
except AmbiguousTargetUnitError:
    traceback.print_exc(limit=0)

# %% [markdown]
# In such a case, we can instead create our desired units ourselves
# and call `convert_unit` or `convert_unit_from_target_series` directly instead.

# %%
desired_units = (
    df_scenarios_differing_units.loc[pix.isin(scenario="scenario_0")]
    .index.droplevel("scenario")
    .to_frame()["unit"]
    .reset_index("unit", drop=True)
)
desired_units

# %%
df_history.openscm.convert_unit(desired_units)

# %%
# The functional equivalent
convert_unit_from_target_series(df_history, desired_units)

# %% [markdown]
# This then makes further operations, like concatenating the two datasets much simpler.

# %%
df_full_timeseries = pd.concat(
    [
        v.dropna(how="all", axis="columns")
        for v in df_history.align(df_scenarios_like_history)
    ],
    axis="columns",
)
df_full_timeseries
