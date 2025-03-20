"""
Tests of `pandas_openscm.plotting` and `pd.DataFrame.openscm.plot*`
"""

from __future__ import annotations

import numpy as np
import pytest

from pandas_openscm.plotting import plot_plume
from pandas_openscm.testing import create_test_df

plt = pytest.importorskip("matplotlib.pyplot")

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


def test_plot_plume(tmp_path, image_regression, setup_pandas_accessor):
    df = create_test_df(
        variables=(("variable_1", "K"),),
        n_scenarios=5,
        n_runs=10,
        timepoints=np.arange(1950.0, 1965.0),
        # TODO: make the rng injectable so we get consistent results each run
    )

    fig, ax = plt.subplots()

    # # TODO: switch to the below
    # df.openscm.plumeplot()
    plot_plume(df, ax=ax)

    out_file = tmp_path / "fig.png"
    plt.savefig(out_file, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


def test_plot_plume_pre_calculated():
    assert False


def test_plot_plume_units():
    # Plot two different sets of data with different units
    # Make sure that the unit handling passes through
    assert False
