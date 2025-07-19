"""
Integration tests of `pandas_openscm.unit_conversion`
"""

from __future__ import annotations

import numpy as np
import pytest

from pandas_openscm.testing import create_test_df
from pandas_openscm.unit_conversion import convert_unit, convert_unit_like


@pytest.mark.parametrize(
    "unit, exp_unit",
    (
        pytest.param(None, "unit", id="default"),
        ("units", "units"),
    ),
)
def test_convert_unit_single_unit(unit, exp_unit):
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

    call_kwargs = {}
    if unit is not None:
        start = (
            start.reset_index("unit")
            .rename({"unit": unit}, axis="columns")
            .set_index(unit, append=True)
        )
        call_kwargs["unit_level"] = unit

    res = convert_unit(start, "K", **call_kwargs)

    assert (res.index.get_level_values(exp_unit) == "K").all()

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
        variables=[
            ("temperature", "K"),
            ("erf", "W / m^2"),
            ("ohc", "ZJ"),
        ],
        n_scenarios=2,
        n_runs=2,
        timepoints=np.array([1850.0, 2000.0, 2050.0, 2100.0]),
    )

    # Don't convert W / m^2
    res = convert_unit(start, {"K": "degC", "ZJ": "J"})

    np.testing.assert_allclose(
        res.loc[res.index.get_level_values("variable") == "temperature", :].values,
        -273.15
        + start.loc[
            start.index.get_level_values("variable") == "temperature", :
        ].values,
    )

    np.testing.assert_equal(
        res.loc[res.index.get_level_values("variable") == "erf", :].values,
        start.loc[start.index.get_level_values("variable") == "erf", :].values,
    )

    np.testing.assert_equal(
        res.loc[res.index.get_level_values("variable") == "ohc", :].values,
        1e21 * start.loc[start.index.get_level_values("variable") == "ohc", :].values,
    )


def test_convert_series():
    # Check that conversion works if user supplies a Series of target units
    start = create_test_df(
        variables=[
            ("temperature", "K"),
            ("erf", "W / m^2"),
            ("ohc", "ZJ"),
        ],
        n_scenarios=2,
        n_runs=2,
        timepoints=np.array([1850.0, 2000.0, 2050.0, 2100.0]),
    )

    target_units = start.reset_index("unit")["unit"].replace(
        {"W / m^2": "ZJ / yr / m^2", "ZJ": "PJ"}
    )

    res = convert_unit(start, target_units)

    np.testing.assert_allclose(
        res.loc[res.index.get_level_values("variable") == "temperature", :].values,
        start.loc[start.index.get_level_values("variable") == "temperature", :].values,
    )

    np.testing.assert_allclose(
        res.loc[res.index.get_level_values("variable") == "erf", :].values,
        (60.0 * 60.0 * 24.0 * 365.25)
        * 1e-21
        * start.loc[start.index.get_level_values("variable") == "erf", :].values,
    )

    np.testing.assert_allclose(
        res.loc[res.index.get_level_values("variable") == "ohc", :].values,
        1e6 * start.loc[start.index.get_level_values("variable") == "ohc", :].values,
    )


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


# To write:
# - no op i.e. what happens when df is already in the right units
# - tests of various error paths
