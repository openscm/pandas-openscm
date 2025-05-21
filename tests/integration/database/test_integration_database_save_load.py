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
    get_parametrized_db_data_backends,
    get_parametrized_db_index_backends,
)


@get_parametrized_db_data_backends()
@get_parametrized_db_index_backends()
def test_save_and_load_basic(tmpdir, db_data_backend, db_index_backend):
    if "Feather" in str(db_data_backend) or "Feather" in str(db_index_backend):
        pytest.importorskip("pyarrow")

    if "netCDF" in str(db_data_backend) or "netCDF" in str(db_index_backend):
        pytest.importorskip("xarray")

    df_timeseries_like = pd.DataFrame(
        np.arange(12).reshape(4, 3),
        columns=pd.Index([2010, 2015, 2025], name="year"),
        index=pd.MultiIndex.from_tuples(
            [
                ("scenario_a", "climate_model_a", "Temperature", "K"),
                ("scenario_b", "climate_model_a", "Temperature", "K"),
                ("scenario_b", "climate_model_b", "Temperature", "K"),
                ("scenario_b", "climate_model_b", "Ocean Heat Uptake", "J"),
            ],
            names=["scenario", "climate_model", "variable", "unit"],
        ),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
        index_file_lock=nullcontext(),  # not used
    )

    db.save(df_timeseries_like)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(df_timeseries_like.index.names)
    pd.testing.assert_index_equal(
        df_timeseries_like.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(
        out_columns_type=df_timeseries_like.columns.dtype,
        out_columns_name=df_timeseries_like.columns.name,
    )

    assert_frame_alike(df_timeseries_like, loaded)

    # Check the file map, index and metadata too
    file_map = db.load_file_map()
    assert file_map.size == 1, "should only be one file"

    index = db.load_index()
    assert index.columns.tolist() == ["file_id"]
    pd.testing.assert_index_equal(
        index.index,
        df_timeseries_like.index.reorder_levels(index.index.names),
        check_order=False,
    )

    metadata = db.load_metadata()
    pd.testing.assert_index_equal(
        metadata,
        df_timeseries_like.index.reorder_levels(metadata.names),
        check_order=False,
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
        index_file_lock=nullcontext(),  # not used
    )

    db.save(start)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(start.index.names)
    pd.testing.assert_index_equal(
        start.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=start.columns.dtype)

    assert_frame_alike(start, loaded)

    # Check the file map, index and metadata too
    file_map = db.load_file_map()
    assert file_map.size == 1, "should only be one file"

    index = db.load_index()
    assert index.columns.tolist() == ["file_id"]
    pd.testing.assert_index_equal(
        index.index, start.index.reorder_levels(index.index.names), check_order=False
    )

    metadata = db.load_metadata()
    pd.testing.assert_index_equal(
        metadata, start.index.reorder_levels(metadata.names), check_order=False
    )


def test_save_multiple_and_load(tmpdir):
    pix = pytest.importorskip("pandas_indexing")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
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
    pix = pytest.importorskip("pandas_indexing")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
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
        index_file_lock=nullcontext(),  # not used
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
    pix = pytest.importorskip("pandas_indexing")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
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
    pix = pytest.importorskip("pandas_indexing")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
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
    warn_on_partial_overwrite,
    expectation_overwrite_warning,
    tmpdir,
):
    pix = pytest.importorskip("pandas_indexing")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
    )

    cdf = partial(
        create_test_df,
        variables=[(f"v_{i}", "m") for i in range(5)],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    original = cdf(n_scenarios=6)
    db.save(original)

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


# Parameterise here because this is so fundamental.
# For rest of checking backend combos,
# rely on state tests and design of the codebase
# (where backend isn't involved in determining groupings for data).
@get_parametrized_db_data_backends()
@get_parametrized_db_index_backends()
def test_save_overwrite_force_half_overlap_all_backends(
    db_data_backend,
    db_index_backend,
    tmpdir,
):
    pix = pytest.importorskip("pandas_indexing")

    if "Feather" in str(db_data_backend) or "Feather" in str(db_index_backend):
        pytest.importorskip("pyarrow")

    if "netCDF" in str(db_data_backend) or "netCDF" in str(db_index_backend):
        pytest.importorskip("xarray")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
        index_file_lock=nullcontext(),  # not used
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

    db.save(original_overwrite, allow_overwrite=True, warn_on_partial_overwrite=False)

    update_exp = pix.concat([original.loc[~overlap_idx], original_overwrite])
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(update_exp.index.names)
    pd.testing.assert_index_equal(
        update_exp.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(update_exp, loaded)


def test_load_with_loc(tmpdir):
    pix = pytest.importorskip("pandas_indexing")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
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
        index_file_lock=nullcontext(),  # not used
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
        index_file_lock=nullcontext(),  # not used
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
    pix = pytest.importorskip("pandas_indexing")

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
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
