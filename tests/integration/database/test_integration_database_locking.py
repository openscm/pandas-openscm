"""
Tests of saving and loading with `pandas_openscm.OpenSCMDB`
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import filelock
import numpy as np
import pytest

from pandas_openscm.db import InMemoryDataBackend, InMemoryIndexBackend, OpenSCMDB
from pandas_openscm.testing import create_test_df


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
def test_locking(tmpdir, meth, args):
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

    # Learning here:
    # - only use acquire as a context manager
    #   until we have an answer in https://github.com/tox-dev/filelock/issues/402
    # - hence only support a reader within a context manager for now
    # - update tests accordingly

    # This behaves
    with db.index_file_lock.acquire():
        with pytest.raises(filelock.Timeout):
            db.index_file_lock.acquire(timeout=0.02)

    # This also behaves
    with db.index_file_lock.acquire():
        with pytest.raises(filelock.Timeout):
            with db.index_file_lock.acquire(timeout=0.02):
                pass

    # This doesn't do what you think,
    # because you basically acquire and release the lock immediately.
    db.index_file_lock.acquire()
    # Hence no raising here
    db.index_file_lock.acquire(timeout=0.02)

    # However, if you keep the lock instance around,
    # then the releasing doesn't happen immediately.
    db_lock = db.index_file_lock
    db_lock.acquire()
    with pytest.raises(filelock.Timeout):
        db.index_file_lock.acquire(timeout=0.02)

    assert False

    with db.index_file_lock.acquire():
        # # If you try and acquire another lock with the same path,
        # # it will time out.
        # filelock.SoftFileLock(db.index_file_lock_path).acquire(timeout=0.1)
        # with filelock.SoftFileLock(db.index_file_lock_path).acquire(timeout=0.1):
        #     pass

        # Using actually the same lock isn't an issue though.
        with db.index_file_lock.acquire(timeout=0.2):
            pass

        assert db.index_file_lock.is_locked

        # Check that we can't acquire the lock in another thread to use the method
        with pytest.raises(filelock.Timeout):
            getattr(db, meth)(
                # Can't use defaults here as default is no timeout
                lock_context_manager=db.index_file_lock.acquire(timeout=0.0),
                *args,
            )

        # Check that we can't re-acquire the lock even if we have a timeout
        with pytest.raises(filelock.Timeout):
            getattr(db, meth)(
                # Can't use defaults here as default is no timeout
                lock_context_manager=db.index_file_lock.acquire(timeout=0.1),
                *args,
            )

        # Check that we can bypyass the lock acquisition
        getattr(db, meth)(lock_context_manager=nullcontext(), *args)


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


def test_lock_is_always_same(tmpdir):
    """
    This explains why acquiring was so confusing

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
