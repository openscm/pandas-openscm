"""
Tests of reading using an in-memory index with `pandas_openscm`
"""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.testing import assert_frame_alike, create_test_df


def test_load_via_reader_context_manager(tmpdir):
    start = create_test_df(
        n_scenarios=10,
        variables=[("a", "kg"), ("b", "kg"), ("c", "kg")],
        n_runs=5,
        timepoints=np.arange(2015.0, 2100.0, 5.0),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=FeatherDataBackend(),
        backend_index=FeatherIndexBackend(),
    )

    db.save(start)

    with db.create_reader() as reader:
        db_metadata = db.load_metadata()
        reader_metadata = reader.metadata

        metadata_compare = reader_metadata.reorder_levels(db_metadata.index.names)
        pd.testing.assert_index_equal(
            db_metadata, metadata_compare, exact="equiv", check_order=False
        )

        loaded = reader.load(out_columns_type=start.columns.dtype)

        assert_frame_alike(start, loaded)


def test_load_via_reader(tmpdir):
    start = create_test_df(
        n_scenarios=10,
        variables=[("a", "kg"), ("b", "kg"), ("c", "kg")],
        n_runs=5,
        timepoints=np.arange(2015.0, 2100.0, 5.0),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=FeatherDataBackend(),
        backend_index=FeatherIndexBackend(),
    )

    db.save(start)

    reader = db.create_reader()

    db_metadata = db.load_metadata()
    reader_metadata = reader.metadata

    metadata_compare = reader_metadata.reorder_levels(db_metadata.index.names)
    pd.testing.assert_index_equal(
        db_metadata, metadata_compare, exact="equiv", check_order=False
    )

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
    start = create_test_df(
        n_scenarios=10,
        variables=[("a", "kg"), ("b", "kg"), ("c", "kg")],
        n_runs=5,
        timepoints=np.arange(2015.0, 2100.0, 5.0),
    )

    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=FeatherDataBackend(),
        backend_index=FeatherIndexBackend(),
    )

    db.save(start)

    # If we use the context manager,
    # the created object holds the lock
    # so we can't read/write with the db anymore.
    # There is deliberately no bypass for this
    # (if you want to bypass, don't use the context manager).
    with db.create_reader():
        with pytest.raises(filelock.Timeout):
            db.index_file_lock.acquire(timeout=0.02)

    # By default, using `create_reader` also holds the lock.
    reader = db.create_reader()

    with pytest.raises(filelock.Timeout):
        db.index_file_lock.acquire(timeout=0.02)

    # So we have to explicitly release the lock.
    reader.release_lock()

    # Now we can get the lock again
    with does_not_raise():
        db.index_file_lock.acquire(timeout=0.02)

    # We can create a reader that isn't holding the lock if we want
    reader = db.create_reader(acquire_lock=False)

    with does_not_raise():
        db.index_file_lock.acquire(timeout=0.02)

    # This is a no op,
    # but this checks that calling again doesn't cause anything to explode
    reader.release_lock()
