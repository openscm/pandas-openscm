"""
Tests of making plans for moving data with `pandas_openscm.OpenSCMDB`
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from pandas_openscm.db import (
    InMemoryDataBackend,
    InMemoryIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.db.rewriting import MovePlan, ReWriteAction, make_move_plan
from pandas_openscm.testing import assert_move_plan_equal


def test_make_move_plan_no_overwrite(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=contextlib.nullcontext(),  # not used
    )

    index_start = pd.DataFrame(
        [
            ("scenario_a", "variable_a", "Mt", 0),
            ("scenario_a", "variable_b", "Mt", 0),
        ],
        columns=["scenario", "variable", "unit", "file_id"],
    ).set_index(["scenario", "variable", "unit"])
    file_map_start = pd.Series(
        [
            db.get_new_data_file_path(fid).rel_db
            for fid in index_start["file_id"].unique()
        ],
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

    res = make_move_plan(
        index_start=index_start,
        file_map_start=file_map_start,
        data_to_write=data_to_write,
        get_new_data_file_path=db.get_new_data_file_path,
        db_dir=db.db_dir,
    )

    assert_move_plan_equal(res, exp)


def test_make_move_plan_full_overwrite(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=contextlib.nullcontext(),  # not used
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
        [
            db.get_new_data_file_path(fid).rel_db
            for fid in index_start["file_id"].unique()
        ],
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
        [db.get_new_data_file_path(file_id).rel_db for file_id in exp_moved_file_ids],
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
        delete_paths=(db.db_dir / file_map_start.loc[1],),
    )

    res = make_move_plan(
        index_start=index_start,
        file_map_start=file_map_start,
        data_to_write=data_to_write,
        get_new_data_file_path=db.get_new_data_file_path,
        db_dir=db.db_dir,
    )

    assert_move_plan_equal(res, exp)


def test_make_move_plan_partial_overwrite(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=contextlib.nullcontext(),  # not used
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
        [
            db.get_new_data_file_path(fid).rel_db
            for fid in index_start["file_id"].unique()
        ],
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
        [db.get_new_data_file_path(file_id).rel_db for file_id in exp_moved_file_ids],
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
                from_file=db.db_dir / file_map_start.loc[2],
                locator=pd.MultiIndex.from_frame(
                    pd.DataFrame(
                        [
                            ("scenario_c", "variable_a", "Mt"),
                        ],
                        columns=["scenario", "variable", "unit"],
                    )
                ),
                to_file=db.db_dir / exp_moved_file_map.loc[3],
            ),
        ),
        delete_paths=(
            db.db_dir / v for v in (file_map_start.loc[1], file_map_start.loc[2])
        ),
    )

    res = make_move_plan(
        index_start=index_start,
        file_map_start=file_map_start,
        data_to_write=data_to_write,
        get_new_data_file_path=db.get_new_data_file_path,
        db_dir=db.db_dir,
    )

    assert_move_plan_equal(res, exp)
