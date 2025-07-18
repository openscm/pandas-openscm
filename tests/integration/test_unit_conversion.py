"""
Integration tests of `pandas_openscm.unit_conversion`
"""

from __future__ import annotations

import numpy as np
import pytest

from pandas_openscm.testing import create_test_df
from pandas_openscm.unit_conversion import convert_unit, convert_unit_like


def test_convert_unit_single_unit():
    start = create_test_df(
        variables=[
            ("Cold", "mK"),
            ("Warm", "kK"),
            ("Body temperature", "degC"),
        ],
        n_scenarios=2,
        n_runs=3,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    res = convert_unit(start, "K")

    assert (res.index.get_level_values("unit") == "K").all()

    np.testing.assert_equal(
        res.loc[res.index.get_level_values("variable") == "Cold", :].values,
        1e-3 * start.loc[start.index.get_level_values("variable") == "Cold", :].values,
    )

    np.testing.assert_equal(
        res.loc[res.index.get_level_values("variable") == "Warm", :].values,
        1e3 * start.loc[start.index.get_level_values("variable") == "Warm", :].values,
    )

    np.testing.assert_equal(
        res.loc[res.index.get_level_values("variable") == "Body temperature", :].values,
        273.15
        + start.loc[
            start.index.get_level_values("variable") == "Body temperature", :
        ].values,
    )


def test_convert_unit_ur_injection():
    pint = pytest.importorskip("pint")

    start = create_test_df(
        variables=[("Wavelength", "m")],
        n_scenarios=2,
        n_runs=2,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    # Without injection, raises
    with pytest.raises(pint.DimensionalityError):
        convert_unit(start, "Hz")

    # With injection and context, all good
    ur = pint.UnitRegistry()
    with ur.context("spectroscopy"):
        res = convert_unit(start, "Hz", ur=ur)

    np.testing.assert_allclose(
        res.values,
        2.998 * 1e8 / start.values,
        rtol=1e-4,
    )


def test_convert_unit_mapping():
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    # Don't convert W / m^2
    res = start.convert_unit({"Mt CO2/yr": "Gt C/yr", "ZJ": "J"})

    np.testing.assert_equal(
        res.iloc[0, :].values,
        start.iloc[0, :].values * 12.0 / 44000.0,
    )

    np.testing.assert_equal(
        res.iloc[1, :].values,
        start.iloc[1, :].values,
    )

    np.testing.assert_equal(
        res.iloc[2, :].values,
        start.iloc[2, :].values * 1e21,
    )


# - test error paths
def test_convert_unit_like():
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    target = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    res = convert_unit_like(start, target)

    np.testing.assert_equal(
        res.iloc[0, :].values,
        start.iloc[0, :].values * 1e3,
    )

    np.testing.assert_equal(
        res.iloc[1, :].values,
        start.iloc[1, :].values / 1e3,
    )

    np.testing.assert_equal(
        res.iloc[2, :].values,
        start.iloc[2, :].values + 273.0,
    )


def test_convert_unit_like_ur_injection():
    pint = pytest.importorskip("pint")

    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    # Without injection, raises
    with pytest.raises(pint.DimensionalityError):
        convert_unit_like(start, target)

    # With injection and context, all good
    ur = pint.UnitRegistry()
    with ur.context("spectroscopy"):
        res = convert_unit_like(start, target)

    np.testing.assert_allclose(
        res.iloc[0, :].values,
        2.998 * 1e8 / start.iloc[0, :].values,
        rtol=1e-4,
    )


def test_convert_unit_like_different_unit_level_infer_target_unit_level():
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    target = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    res = convert_unit_like(start, target, df_unit_level)

    np.testing.assert_equal(
        res.iloc[0, :].values,
        start.iloc[0, :].values * 1e3,
    )

    np.testing.assert_equal(
        res.iloc[1, :].values,
        start.iloc[1, :].values / 1e3,
    )

    np.testing.assert_equal(
        res.iloc[2, :].values,
        start.iloc[2, :].values + 273.0,
    )


def test_convert_unit_like_different_unit_level_explicit_target_level():
    start = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    target = create_test_df(
        variables=[(f"variable_{i}", "Mt") for i in range(5)],
        n_scenarios=3,
        n_runs=6,
        timepoints=np.array([1.0, 2.0, 3.0]),
    )

    res = convert_unit_like(start, target, df_unit_level, target_unit_level)

    np.testing.assert_equal(
        res.iloc[0, :].values,
        start.iloc[0, :].values * 1e3,
    )

    np.testing.assert_equal(
        res.iloc[1, :].values,
        start.iloc[1, :].values / 1e3,
    )

    np.testing.assert_equal(
        res.iloc[2, :].values,
        start.iloc[2, :].values + 273.0,
    )
