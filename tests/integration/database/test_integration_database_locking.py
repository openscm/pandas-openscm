"""
Tests of saving and loading with `pandas_openscm.OpenSCMDB`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pandas_openscm.db import InMemoryDataBackend, InMemoryIndexBackend, OpenSCMDB
from pandas_openscm.testing import create_test_df

filelock = pytest.importorskip("filelock")


def test_lock_is_always_same(tmpdir):
    """
    Test that each instance has its own lock and it is the same on each access

    (This may seem like an odd test, but a previous implementation failed this,
    which led to super weird behaviour.)

    In future, maybe this can be removed,
    but I'd be careful about failing this test.
    If you get the lock, for a given instance,
    it should always be the same thing.
    """
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    access_one = db.index_file_lock
    access_two = db.index_file_lock

    assert access_one == access_two


def test_lock_acquisition(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    db.index_file_lock.acquire()
    assert db.index_file_lock.is_locked
    db.index_file_lock.release()
    assert not db.index_file_lock.is_locked


def test_acquire_lock_twice(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    db.index_file_lock.acquire()
    db.index_file_lock.acquire().lock.acquire(timeout=1.0)

    with db.index_file_lock.acquire():
        db.index_file_lock.acquire(timeout=1.0)


@pytest.mark.parametrize(
    "meth, kwargs",
    (
        ("delete", {}),
        ("load", {}),
        ("load_file_map", {}),
        ("load_metadata", {}),
        (
            "save",
            dict(
                data=create_test_df(
                    variables=(("variable", "kg"),),
                    n_scenarios=1,
                    n_runs=1,
                    timepoints=np.array([1.0, 1.5]),
                ),
                allow_overwrite=True,
            ),
        ),
    ),
)
def test_locking_multi_step(tmpdir, meth, kwargs):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
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

    db.index_file_lock.acquire()
    assert db.index_file_lock.is_locked
    # You can acquire the same lock again
    db.index_file_lock.acquire()
    # But if you try to create a new lock,
    # this won't work (either in the same thread of a different one)
    with pytest.raises(filelock.Timeout):
        filelock.FileLock(db.index_file_lock_path).acquire(timeout=0.02)

    # Same logic as above: you can execute methods
    # with the same lock
    getattr(db, meth)(**kwargs)
    with pytest.raises(filelock.Timeout):
        # But a different lock will fail.
        getattr(db, meth)(
            index_file_lock=filelock.FileLock(db.index_file_lock_path, timeout=0.02),
            **kwargs,
        )

    # If we release the lock, now the other paths do work.
    # We have to release the lock twice as we acquired it twice above.
    db.index_file_lock.release()
    db.index_file_lock.release()
    assert not db.index_file_lock.is_locked

    # Can releaes and acquire
    new_lock = filelock.FileLock(db.index_file_lock_path)
    new_lock.acquire(timeout=0.02)
    new_lock.release()
    # Can run methods
    getattr(db, meth)(
        index_file_lock=filelock.FileLock(db.index_file_lock_path, timeout=0.02),
        **kwargs,
    )

    # The same logic applies with context managers
    with db.index_file_lock.acquire():
        assert db.index_file_lock.is_locked
        # Acquiring the same lock again is fine
        db.index_file_lock.acquire()
        # Release again here to avoid double locking
        db.index_file_lock.release()
        # If you try to create a new lock, this won't work.
        with pytest.raises(filelock.Timeout):
            filelock.FileLock(db.index_file_lock_path).acquire(timeout=0.02)

        # You can execute methods that use the instance's lock
        getattr(db, meth)(**kwargs)
        # But a different lock will fail.
        with pytest.raises(filelock.Timeout):
            getattr(db, meth)(
                # Can't use defaults here as default is no timeout
                index_file_lock=(
                    filelock.FileLock(db.index_file_lock_path, timeout=0.02)
                ),
                **kwargs,
            )

    # Out of the context manager, the lock is released
    assert not db.index_file_lock.is_locked
    filelock.FileLock(db.index_file_lock_path).acquire(timeout=0.02)
    getattr(db, meth)(
        # Can't use defaults here as default is no timeout
        index_file_lock=filelock.FileLock(db.index_file_lock_path, timeout=0.02),
        **kwargs,
    )
