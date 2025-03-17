"""
Tests of reading using an in-memory index with `pandas_openscm`
"""

from __future__ import annotations

import contextlib
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import (
    CSVDataBackend,
    CSVIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.testing import assert_frame_alike, create_test_df


def check_metadata_load_is_same(
    reader_metadata: pd.MultiIndex, db_metadata: pd.MultiIndex
) -> None:
    metadata_compare = reader_metadata.reorder_levels(db_metadata.names)
    pd.testing.assert_index_equal(
        db_metadata, metadata_compare, exact="equiv", check_order=False
    )


def test_load_via_reader_context_manager(tmpdir):
    pytest.importorskip("filelock")

    start = create_test_df(
        n_scenarios=10,
        variables=[("a", "kg"), ("b", "kg"), ("c", "kg")],
        n_runs=5,
        timepoints=np.arange(2015.0, 2100.0, 5.0),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=CSVDataBackend(),
        backend_index=CSVIndexBackend(),
    )

    db.save(start)

    db_metadata = db.load_metadata()

    with db.create_reader() as reader:
        assert reader.lock.is_locked

        check_metadata_load_is_same(reader.metadata, db_metadata)

        loaded = reader.load(out_columns_type=start.columns.dtype)
        assert_frame_alike(start, loaded)

    # Lock released on exiting the context block
    assert not reader.lock.is_locked


def test_load_via_reader(tmpdir):
    start = create_test_df(
        n_scenarios=10,
        variables=[("a", "kg"), ("b", "kg"), ("c", "kg")],
        n_runs=5,
        timepoints=np.arange(2015.0, 2100.0, 5.0),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=CSVDataBackend(),
        backend_index=CSVIndexBackend(),
        index_file_lock=contextlib.nullcontext(),  # not used
    )

    db.save(start)

    reader = db.create_reader(lock=False)

    db_metadata = db.load_metadata()
    check_metadata_load_is_same(reader.metadata, db_metadata)

    loaded = reader.load(out_columns_type=start.columns.dtype)

    assert_frame_alike(start, loaded)


def test_reader_locking(tmpdir):
    """
    Test the handling of locking via the reader

    This could be split out to provide better diagnostic power of issues.

    However, this overall flow is also a handy integration test
    so I would suggest keeping this test as is
    and just adding more tests
    (at least until maintaining this test becomes annoying).
    """
    filelock = pytest.importorskip("filelock")

    start = create_test_df(
        n_scenarios=10,
        variables=[("a", "kg"), ("b", "kg"), ("c", "kg")],
        n_runs=5,
        timepoints=np.arange(2015.0, 2100.0, 5.0),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=CSVDataBackend(),
        backend_index=CSVIndexBackend(),
    )

    db.save(start)

    # If we use the context manager,
    # the created object holds the lock
    # so we can't read/write with the db anymore.
    with db.create_reader() as reader:
        # The db and the reader get different locks by default
        assert db.index_file_lock != reader.lock
        # The db's lock isn't locked
        assert not db.index_file_lock.is_locked
        # The reader's lock is locked
        assert reader.lock.is_locked

        with pytest.raises(filelock.Timeout):
            # We can't acquire the db's lock,
            # because the reader has the lock.
            db.index_file_lock.acquire(timeout=0.02)

    # Once we're out of the context block, the lock is released
    with does_not_raise():
        db.index_file_lock.acquire(timeout=0.02)

    db.index_file_lock.release()

    # You can bypass holding the lock within the context manager
    # (doesn't make much sense, just don't use the context manager in this case,
    # but at least it doesn't explode).
    with db.create_reader(lock=False) as reader:
        assert reader.lock is None

        with does_not_raise():
            db.index_file_lock.acquire(timeout=0.02)

        db.index_file_lock.release()

    # By default, using `create_reader` does not hold the lock.
    reader = db.create_reader()

    # So we could keep doing database ops
    db.index_file_lock.acquire()
    # (Release again to avoid double releasing later)
    db.index_file_lock.release()

    # However, if we now acquire the reader's lock
    reader.lock.acquire()
    # The db is locked again
    with pytest.raises(filelock.Timeout):
        db.index_file_lock.acquire(timeout=0.02)

    # If we release the lock.
    reader.lock.release()

    # We can get the lock again
    with does_not_raise():
        db.index_file_lock.acquire(timeout=0.02)

    # This is a no op,
    # but this checks that calling again doesn't cause anything to explode
    reader.lock.release()

    # We can also create the reader within a context block that is managing the lock.
    # We do have to be a bit more careful in this case.
    with db.index_file_lock as lock:
        # By default, everything locks
        # because the reader is trying to acquire and it can't.
        # with db.create_reader() as reader:
        #     ...
        # This code illustrates the above without making the test hang forever.
        with pytest.raises(filelock.Timeout):
            with db.create_reader(
                lock=filelock.FileLock(db.index_file_lock_path, timeout=0.02)
            ) as reader:
                ...

        # If we pass in the lock we're already managing, then there is no issue
        # and we can get information from either path.
        with db.create_reader(lock=lock) as reader:
            check_metadata_load_is_same(reader.metadata, db.load_metadata())
            assert_frame_alike(reader.load(), db.load())
