"""
Tests of `pandas_openscm.OpenSCMDB`
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pytest

from pandas_openscm.db import (
    AlreadyInDBError,
    CSVDataBackend,
    CSVIndexBackend,
    EmptyDBError,
    MovePlan,
    OpenSCMDB,
    ReWriteAction,
)
from pandas_openscm.testing import (
    assert_frame_alike,
    assert_move_plan_equal,
    create_test_df,
    get_parametrized_db_data_backends,
    get_parametrized_db_index_backends,
)

db_data_backends = get_parametrized_db_data_backends()
db_index_backends = get_parametrized_db_index_backends()


@pytest.mark.parametrize(
    "n_scenarios, variables, n_runs",
    (
        pytest.param(2, [("a", "kg"), ("b", "kg"), ("c", "kg")], 4, id="small"),
        pytest.param(
            20,
            [("a", "g"), ("b", "m"), ("c", "s")],
            60,
            id="medium",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            100,
            [("a", "g"), ("b", "m"), ("c", "s")],
            600,
            id="large",
            marks=pytest.mark.slow,
        ),
        # # Blows some integer limit, to turn back on in future
        # pytest.param(
        #     1000,
        #     [("a", "kg"), ("b", "W / m^2"), ("c", "yr")],
        #     600,
        #     id="x-arge",
        #     marks=pytest.mark.slow,
        # ),
    ),
)
@db_data_backends
@db_index_backends
def test_save_and_load(  # noqa: PLR0913
    n_scenarios, variables, n_runs, db_data_backend, db_index_backend, tmpdir
):
    if (db_data_backend == CSVDataBackend or db_index_backend == CSVIndexBackend) and (
        n_scenarios * len(variables) * n_runs
    ) > 25000:
        pytest.skip("Too slow")

    start = create_test_df(
        n_scenarios=n_scenarios,
        variables=variables,
        n_runs=n_runs,
        timepoints=np.arange(1750, 2100),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
    )

    db.save(start)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(start.index.names)
    pd.testing.assert_index_equal(
        start.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=int)

    assert_frame_alike(start, loaded)


@db_data_backends
@db_index_backends
def test_save_multiple_and_load(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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


@db_data_backends
@db_index_backends
def test_save_multiple_grouped_and_load(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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


@db_data_backends
@db_index_backends
def test_save_overwrite_error(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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


@db_data_backends
@db_index_backends
def test_save_overwrite_force(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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


@db_data_backends
@db_index_backends
def test_save_overwrite_force_half_overlap(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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
    # TODO: make warn_on_partial_overwrite a parameter and test
    db.save(original_overwrite, allow_overwrite=True, warn_on_partial_overwrite=False)

    update_exp = pix.concat([original.loc[~overlap_idx], original_overwrite])
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata.reorder_levels(update_exp.index.names)
    pd.testing.assert_index_equal(
        update_exp.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    assert_frame_alike(update_exp, loaded)


@pytest.mark.parametrize(
    "meth, args",
    (
        ("delete", []),
        ("load", []),
        ("load_file_map", []),
        ("load_metadata", []),
        (
            "save",
            [
                create_test_df(
                    variables=(("variable", "kg"),),
                    n_scenarios=1,
                    n_runs=1,
                    timepoints=np.array([1.0, 1.5]),
                )
            ],
        ),
    ),
)
@db_data_backends
@db_index_backends
def test_locking(tmpdir, meth, args, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
    )

    # Put some data in the db so there's something to lock
    db.save(
        create_test_df(
            n_scenarios=1,
            variables=[("a", "K")],
            n_runs=1,
            timepoints=np.array([10.0, 15.0]),
        )
    )

    # Acquire the lock
    with db.index_file_lock:
        # Check that we can't re-acquire the lock to use the method
        with pytest.raises(filelock.Timeout):
            getattr(db, meth)(
                # Can't use defaults here as default is no timeout
                lock_context_manager=db.index_file_lock.acquire(timeout=0.0),
                *args,
            )

        with pytest.raises(filelock.Timeout):
            getattr(db, meth)(
                # Can't use defaults here as default is no timeout
                lock_context_manager=db.index_file_lock.acquire(timeout=0.1),
                *args,
            )

        # Unless we pass in a different context manager
        getattr(db, meth)(lock_context_manager=nullcontext(), *args)


@db_data_backends
@db_index_backends
def test_load_with_loc(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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


@db_data_backends
@db_index_backends
def test_load_with_index_all(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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
@db_data_backends
@db_index_backends
def test_load_with_index_slice(tmpdir, slice, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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
@db_data_backends
@db_index_backends
def test_load_with_pix_unique_levels(tmpdir, levels, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
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


@db_data_backends
@db_index_backends
def test_deletion(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
    )

    db.save(
        create_test_df(
            n_scenarios=10,
            variables=[("variable_1", "kg"), ("variable_2", "Mt"), ("variable_3", "m")],
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        )
    )

    assert isinstance(db.load(), pd.DataFrame)

    db.delete()

    with pytest.raises(EmptyDBError):
        db.load_metadata()

    with pytest.raises(EmptyDBError):
        db.load()


@db_data_backends
@db_index_backends
def test_make_move_plan_no_overwrite(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
    )

    index_start = pd.DataFrame(
        [
            ("scenario_a", "variable_a", "Mt", 0),
            ("scenario_a", "variable_b", "Mt", 0),
        ],
        columns=["scenario", "variable", "unit", "file_id"],
    ).set_index(["scenario", "variable", "unit"])
    file_map_start = pd.Series(
        [db.get_new_data_file_path(fid) for fid in index_start["file_id"].unique()],
        index=pd.Index(index_start["file_id"].unique(), name="file_id"),
    )

    index_data_to_write = pd.MultiIndex.from_tuples(
        [
            ("scenario_b", "variable_a", "Mt"),
            ("scenario_b", "variable_b", "Mt"),
        ],
        names=["scenario", "variable", "unit"],
    )
    data_to_write = pd.DataFrame(
        np.random.default_rng().random((index_data_to_write.shape[0], 3)),
        index=index_data_to_write,
        columns=np.arange(2020.0, 2023.0),
    )

    # No overlap so no need to move anything,
    # the index and file map are just the same as what we started with
    # (the layer make_move_plan above deals with writing the new data).
    exp = MovePlan(
        moved_index=index_start,
        moved_file_map=file_map_start,
        rewrite_actions=None,
        delete_paths=None,
    )

    res = db.make_move_plan(index_start, file_map_start, data_to_write)

    assert_move_plan_equal(res, exp)


@db_data_backends
@db_index_backends
def test_make_move_plan_full_overwrite(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
    )

    index_start = pd.DataFrame(
        [
            ("scenario_a", "variable_a", "Mt", 0),
            ("scenario_a", "variable_b", "Mt", 0),
            ("scenario_b", "variable_a", "Mt", 1),
            ("scenario_b", "variable_b", "Mt", 1),
        ],
        columns=["scenario", "variable", "unit", "file_id"],
    ).set_index(["scenario", "variable", "unit"])
    file_map_start = pd.Series(
        [db.get_new_data_file_path(fid) for fid in index_start["file_id"].unique()],
        index=pd.Index(index_start["file_id"].unique(), name="file_id"),
    )

    index_data_to_write = pd.MultiIndex.from_tuples(
        [
            # Full overwrite of file 1
            ("scenario_b", "variable_a", "Mt"),
            ("scenario_b", "variable_b", "Mt"),
        ],
        names=["scenario", "variable", "unit"],
    )
    data_to_write = pd.DataFrame(
        np.random.default_rng().random((index_data_to_write.shape[0], 3)),
        index=index_data_to_write,
        columns=np.arange(2020.0, 2023.0),
    )

    exp_moved_file_ids = [0]  # 1 will be overwritten i.e. schedule to delete
    exp_moved_file_map = pd.Series(
        [db.get_new_data_file_path(file_id) for file_id in exp_moved_file_ids],
        index=pd.Index(exp_moved_file_ids, name="file_id"),
    )

    exp_moved_index = pd.DataFrame(
        [
            # Unchanged
            ("scenario_a", "variable_a", "Mt", 0),
            ("scenario_a", "variable_b", "Mt", 0),
            # # Will be overwritten hence deleted
            # ("scenario_b", "variable_a", "Mt", 1),
            # ("scenario_b", "variable_b", "Mt", 1),
        ],
        columns=["scenario", "variable", "unit", "file_id"],
    ).set_index(index_start.index.names)

    exp = MovePlan(
        moved_index=exp_moved_index,
        moved_file_map=exp_moved_file_map,
        rewrite_actions=None,
        delete_paths=(file_map_start.loc[1],),
    )

    res = db.make_move_plan(index_start, file_map_start, data_to_write)

    assert_move_plan_equal(res, exp)


@db_data_backends
@db_index_backends
def test_make_move_plan_partial_overwrite(tmpdir, db_data_backend, db_index_backend):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=db_data_backend(),
        backend_index=db_index_backend(),
    )

    index_start = pd.DataFrame(
        [
            ("scenario_a", "variable_a", "Mt", 0),
            ("scenario_a", "variable_b", "Mt", 0),
            ("scenario_b", "variable_a", "Mt", 1),
            ("scenario_b", "variable_b", "Mt", 1),
            ("scenario_c", "variable_a", "Mt", 2),
            ("scenario_c", "variable_b", "Mt", 2),
        ],
        columns=["scenario", "variable", "unit", "file_id"],
    ).set_index(["scenario", "variable", "unit"])
    file_map_start = pd.Series(
        [db.get_new_data_file_path(fid) for fid in index_start["file_id"].unique()],
        index=pd.Index(index_start["file_id"].unique(), name="file_id"),
    )

    index_data_to_write = pd.MultiIndex.from_tuples(
        [
            # File 0 should be left alone
            # ("scenario_a", "variable_a", "Mt"),
            # ("scenario_a", "variable_b", "Mt"),
            # File 1 should be fully deleted to make room
            # for this data
            ("scenario_b", "variable_a", "Mt"),
            ("scenario_b", "variable_b", "Mt"),
            # File 2 should be partially re-written,
            # keeping variable_a but not variable_b
            # (which will be overwritten)
            # ("scenario_c", "variable_a", "Mt"),
            ("scenario_c", "variable_b", "Mt"),
        ],
        names=["scenario", "variable", "unit"],
    )
    data_to_write = pd.DataFrame(
        np.random.default_rng().random((index_data_to_write.shape[0], 3)),
        index=index_data_to_write,
        columns=np.arange(2020.0, 2023.0),
    )

    exp_moved_file_ids = [0, 3]  # 1 deleted, 2 re-written then deleted
    exp_moved_file_map = pd.Series(
        [db.get_new_data_file_path(file_id) for file_id in exp_moved_file_ids],
        index=pd.Index(exp_moved_file_ids, name="file_id"),
    )

    exp_moved_index = pd.DataFrame(
        [
            # Unchanged
            ("scenario_a", "variable_a", "Mt", 0),
            ("scenario_a", "variable_b", "Mt", 0),
            # # Overwritten
            # ("scenario_b", "variable_a", "Mt", 1),
            # ("scenario_b", "variable_b", "Mt", 1),
            # Re-written to make space
            ("scenario_c", "variable_a", "Mt", 3),
            # # Overwritten
            # ("scenario_c", "variable_b", "Mt", 2),
        ],
        columns=["scenario", "variable", "unit", "file_id"],
    ).set_index(index_start.index.names)

    exp = MovePlan(
        moved_index=exp_moved_index,
        moved_file_map=exp_moved_file_map,
        rewrite_actions=(
            ReWriteAction(
                from_file=file_map_start.loc[2],
                locator=pd.MultiIndex.from_frame(
                    pd.DataFrame(
                        [
                            ("scenario_c", "variable_a", "Mt"),
                        ],
                        columns=["scenario", "variable", "unit"],
                    )
                ),
                to_file=exp_moved_file_map.loc[3],
            ),
        ),
        delete_paths=(file_map_start.loc[1], file_map_start.loc[2]),
    )

    res = db.make_move_plan(index_start, file_map_start, data_to_write)

    assert_move_plan_equal(res, exp)
