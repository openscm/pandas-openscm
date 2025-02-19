"""
Tests of saving and loading with `pandas_openscm.OpenSCMDB`

Note that these are also supplemented by our state testing with hypothesis
(`tests/integration/database/test_integration_database_state.py`),
hence we don't have to test every combination here.
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pytest

from pandas_openscm.db import (
    AlreadyInDBError,
    InMemoryDataBackend,
    InMemoryIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.index_manipulation import unify_index_levels
from pandas_openscm.testing import (
    assert_frame_alike,
    create_test_df,
)


def test_save_and_load(tmpdir):
    start = create_test_df(
        n_scenarios=10,
        variables=[("a", "kg"), ("b", "kg"), ("c", "kg")],
        n_runs=5,
        timepoints=np.arange(2015.0, 2100.0, 5.0),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    db.save(start)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(start.index.names)
    pd.testing.assert_index_equal(
        start.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=start.columns.dtype)

    assert_frame_alike(start, loaded)


def test_save_multiple_and_load(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    all_saved_l = []
    for variable in [
        [("Emissions", "Gt")],
        [("Concentrations", "ppm")],
        [("Forcing", "W/m^2")],
    ]:
        to_save = create_test_df(
            n_scenarios=10,
            variables=variable,
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        )

        db.save(to_save)
        all_saved_l.append(to_save)

    all_saved = pix.concat(all_saved_l)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(all_saved.index.names)
    pd.testing.assert_index_equal(
        all_saved.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(all_saved, loaded)


def test_save_multiple_grouped_and_load(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    all_saved_l = []
    for variable in [
        [("Emissions", "Gt")],
        [("Concentrations", "ppm")],
        [("Forcing", "W/m^2")],
    ]:
        to_save = create_test_df(
            n_scenarios=10,
            variables=variable,
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        )

        db.save(to_save, groupby=["scenario", "variable"])
        all_saved_l.append(to_save)

    all_saved = pix.concat(all_saved_l)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(all_saved.index.names)
    pd.testing.assert_index_equal(
        all_saved.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(all_saved, loaded)


@pytest.mark.parametrize(
    "wide_first",
    (
        pytest.param(True, id="wide-first"),
        pytest.param(False, id="narrow-first"),
    ),
)
def test_save_multiple_grouped_wide_and_narrow_and_load(wide_first, tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    to_save_wide_index = create_test_df(
        n_scenarios=3,
        variables=[("Emission", "Gt"), ("Concentrations", "ppm")],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    to_save_narrow_index = to_save_wide_index.groupby(
        ["variable", "unit", "run"]
    ).mean()
    assert len(to_save_narrow_index.index.names) < len(to_save_wide_index.index.names)

    if wide_first:
        db.save(to_save_wide_index.copy(), groupby=["scenario", "variable"])
        db.save(to_save_narrow_index.copy(), groupby=["variable", "unit"])
    else:
        db.save(to_save_narrow_index.copy(), groupby=["variable", "unit"])
        db.save(to_save_wide_index.copy(), groupby=["scenario", "variable"])

    tmp = unify_index_levels(to_save_wide_index.index, to_save_narrow_index.index)[1]
    to_save_narrow_index_unified_index = to_save_narrow_index.copy()
    to_save_narrow_index_unified_index.index = tmp
    all_saved_exp = pd.concat([to_save_wide_index, to_save_narrow_index_unified_index])

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(all_saved_exp.index.names)
    pd.testing.assert_index_equal(
        all_saved_exp.index, metadata_compare, check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(all_saved_exp, loaded)


def test_save_overwrite_error(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    dup = cdf(variables=[("Emissions", "t")])
    db.save(dup)

    to_save = pix.concat([dup, cdf(variables=[("Weight", "kg")])])

    error_msg = re.escape(
        "The following rows are already in the database:\n"
        f"{dup.index.to_frame(index=False)}"
    )
    with pytest.raises(AlreadyInDBError, match=error_msg):
        db.save(to_save)


def test_save_overwrite_force(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    original = cdf(variables=[("Emissions", "t")])
    db.save(original)

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(original.index.names)
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(original, loaded)

    original_overwrite = cdf(variables=[("Emissions", "t")])
    updated = pix.concat([original_overwrite, cdf(variables=[("Height", "m")])])

    # With force, we can overwrite
    db.save(updated, allow_overwrite=True)

    # Check that the data was overwritten with new data
    try:
        assert_frame_alike(original, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same DataFrame,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(updated.index.names)
    pd.testing.assert_index_equal(
        updated.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(updated, loaded)


@pytest.mark.parametrize(
    "warn_on_partial_overwrite,expectation_overwrite_warning",
    (
        pytest.param(
            None,
            pytest.warns(
                match="Overwriting the data will require re-writing. This may be slow."
            ),
            id="default",
        ),
        pytest.param(
            True,
            pytest.warns(
                match="Overwriting the data will require re-writing. This may be slow."
            ),
            id="explicitly-enabled",
        ),
        pytest.param(False, nullcontext(), id="silenced"),
    ),
)
def test_save_overwrite_force_half_overlap(
    warn_on_partial_overwrite, expectation_overwrite_warning, tmpdir
):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    cdf = partial(
        create_test_df,
        variables=[(f"v_{i}", "m") for i in range(5)],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    original = cdf(n_scenarios=6)
    db.save(original)

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(original.index.names)
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(original, loaded)

    original_overwrite = cdf(n_scenarios=3)

    # Check that the data was overwritten with new data
    overlap_idx = original.index.isin(original_overwrite.index)
    overlap = original.loc[overlap_idx]
    try:
        assert_frame_alike(overlap, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same values,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    # With force, we can overwrite
    call_kwargs = {}
    if warn_on_partial_overwrite is not None:
        call_kwargs["warn_on_partial_overwrite"] = warn_on_partial_overwrite

    with expectation_overwrite_warning:
        db.save(original_overwrite, allow_overwrite=True, **call_kwargs)

    update_exp = pix.concat([original.loc[~overlap_idx], original_overwrite])
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(update_exp.index.names)
    pd.testing.assert_index_equal(
        update_exp.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(update_exp, loaded)


def test_load_with_loc(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    full_db = create_test_df(
        n_scenarios=10,
        variables=[("variable_1", "kg"), ("variable_2", "Mt"), ("variable_3", "m")],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    db.save(full_db, groupby=["scenario"])

    for selector in [
        pix.isin(scenario=["scenario_1", "scenario_3"]),
        pix.isin(scenario=["scenario_1", "scenario_3"], variable=["variable_2"]),
        (
            pix.isin(scenario=["scenario_1", "scenario_3"])
            & pix.ismatch(variable=["variable_1*"])
        ),
        pix.isin(scenario=["scenario_1", "scenario_3"], variable=["variable_2"]),
    ]:
        loaded = db.load(selector, out_columns_type=float)
        exp = full_db.loc[selector]

        assert_frame_alike(loaded, exp)


def test_load_with_index_all(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    full_db = create_test_df(
        n_scenarios=10,
        variables=[("variable_1", "kg"), ("variable_2", "Mt"), ("variable_3", "m")],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    db.save(full_db, groupby=["scenario"])

    idx = full_db.index
    exp = full_db

    loaded = db.load(idx, out_columns_type=float)

    assert_frame_alike(loaded, exp)


@pytest.mark.parametrize(
    "slice",
    (slice(None, None, None), slice(None, 3, None), slice(2, 4, None), slice(1, 15, 2)),
)
def test_load_with_index_slice(tmpdir, slice):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    full_db = create_test_df(
        n_scenarios=10,
        variables=[("variable_1", "kg"), ("variable_2", "Mt"), ("variable_3", "m")],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    db.save(full_db, groupby=["scenario"])

    idx = full_db.index[slice]
    exp = full_db[slice]

    loaded = db.load(idx, out_columns_type=float)

    assert_frame_alike(loaded, exp)


@pytest.mark.parametrize(
    "levels",
    (
        pytest.param(["scenario"], id="first_level"),
        pytest.param(["variable"], id="not_first_level"),
        pytest.param(["scenario", "variable"], id="multi_level_in_order"),
        pytest.param(["scenario", "variable"], id="multi_level_non_adjacent"),
        pytest.param(["variable", "scenario"], id="multi_level_out_of_order"),
        pytest.param(["run", "variable"], id="multi_level_out_of_order_not_first"),
    ),
)
def test_load_with_pix_unique_levels(tmpdir, levels):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    full_db = create_test_df(
        n_scenarios=10,
        variables=[("variable_1", "kg"), ("variable_2", "Mt"), ("variable_3", "m")],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    db.save(full_db, groupby=["scenario"])

    locator = None
    for level in levels:
        if locator is None:
            locator = pix.isin(**{level: full_db.pix.unique(level)[:2]})
        else:
            locator &= pix.isin(**{level: full_db.pix.unique(level)[:2]})

    exp = full_db.loc[locator]
    idx = exp.pix.unique(levels)

    loaded = db.load(idx, out_columns_type=float)

    assert_frame_alike(loaded, exp)
