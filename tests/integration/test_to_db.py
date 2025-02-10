"""
Tests of `pd.DataFrame.openscm.to_db`
"""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pytest

from pandas_openscm.db import AlreadyInDBError, CSVBackend, OpenSCMDB
from pandas_openscm.testing import create_test_df, get_parametrized_db_backends

db_backends = get_parametrized_db_backends()


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
@db_backends
def test_save_and_load(n_scenarios, variables, n_runs, db_backend, tmpdir):
    if db_backend == CSVBackend and (n_scenarios * len(variables) * n_runs) > 25000:
        pytest.skip("Too slow")

    start = create_test_df(
        n_scenarios=n_scenarios,
        variables=variables,
        n_runs=n_runs,
        timepoints=np.arange(1750, 2100),
    )

    db = OpenSCMDB(db_dir=Path(tmpdir), backend=db_backend())

    # start.openscm.to_db(db)
    db.save(start)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        start.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=int, progress=True)

    pd.testing.assert_frame_equal(start, loaded)


@db_backends
def test_save_multiple_and_load(tmpdir, db_backend):
    db = OpenSCMDB(db_dir=Path(tmpdir), backend=db_backend())

    all_saved_l = []
    for variable in [
        ("Emissions", "Gt"),
        ("Concentrations", "ppm"),
        ("Forcing", "W/m^2"),
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
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        all_saved.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(all_saved, loaded)


@db_backends
def test_save_overwrite_error(tmpdir, db_backend):
    db = OpenSCMDB(db_dir=Path(tmpdir), backend=db_backend())

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


@db_backends
def test_save_overwrite_force(tmpdir, db_backend):
    db = OpenSCMDB(db_dir=Path(tmpdir), backend=db_backend())

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
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(original, loaded)

    original_overwrite = cdf(variables=[("Emissions", "t")])
    updated = pix.concat([original_overwrite, cdf(variables=[("Height", "m")])])

    # With force, we can overwrite
    db.save(updated, allow_overwrite=True)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the new file, but not the original file.
    db_files = list(db.db_dir.glob(f"*{db.backend.ext}"))
    assert set([f.name for f in db_files]) == set(
        f"{prefix}{db.backend.ext}" for prefix in ["1", "index", "filemap"]
    )

    # Check that the data was overwritten with new data
    try:
        pd.testing.assert_frame_equal(original, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same DataFrame,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        updated.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(updated, loaded)


@db_backends
def test_save_overwrite_force_half_overlap(tmpdir, db_backend):
    db = OpenSCMDB(db_dir=Path(tmpdir), backend=db_backend())

    cdf = partial(
        create_test_df,
        variables=[(f"v_{i}", "m") for i in range(5)],
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    original = cdf(n_scenarios=6)
    db.save(original)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the file map file plus written file.
    db_files = list(db.db_dir.glob(f"*{db.backend.ext}"))
    assert set([f.name for f in db_files]) == set(
        f"{prefix}{db.backend.ext}" for prefix in ["0", "index", "filemap"]
    )

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(original, loaded)

    original_overwrite = cdf(n_scenarios=3)

    # With force, we can overwrite
    db.save(original_overwrite, allow_overwrite=True)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the file map file plus the newly written file
    # plus the re-written data file
    # (to handle the need to split the original data so we can keep only what we need),
    # but not the original file.
    db_files = list(db.db_dir.glob(f"*{db.backend.ext}"))
    assert set([f.name for f in db_files]) == set(
        f"{prefix}{db.backend.ext}" for prefix in ["1", "2", "index", "filemap"]
    )

    # Check that the data was overwritten with new data
    overlap_idx = original.index.isin(original_overwrite.index)
    overlap = original.loc[overlap_idx]
    try:
        pd.testing.assert_frame_equal(overlap, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same values,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    update_exp = pix.concat([original.loc[~overlap_idx], original_overwrite])
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        update_exp.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(update_exp, loaded)
