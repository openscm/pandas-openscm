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
from pandas_openscm.testing import (
    create_test_df,
)


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

    # Acquire the lock
    with db.index_file_lock:
        # Check that we can't re-acquire the lock to use the method
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
