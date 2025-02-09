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

from pandas_openscm.testing import create_test_df, get_parametrized_db_formats

db_formats = get_parametrized_db_formats()


@pytest.mark.parametrize(
    "n_scenarios, n_variables, n_runs",
    (
        pytest.param(2, 3, 4, id="small"),
        pytest.param(20, 15, 60, id="medium"),
        pytest.param(20, 15, 600, id="large"),
        # Blows some integer limit
        # pytest.param(100, 15, 600, id="x-large"),
    ),
)
@db_formats
def test_save_and_load(n_scenarios, n_variables, n_runs, db_format, tmpdir):
    if db_format == GCDBDataFormat.CSV and (n_scenarios * n_variables * n_runs) > 25000:
        pytest.skip("Too slow")

    start = create_test_df(
        n_scenarios=n_scenarios,
        n_variables=n_variables,
        n_runs=n_runs,
        timepoints=np.arange(1750, 2100),
        units="Mt",
    )

    db = GCDB(tmpdir, format=db_format)

    db.save(start)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        start.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=int, progress=True)

    pd.testing.assert_frame_equal(start, loaded)


@db_formats
def test_save_multiple_and_load(tmpdir, db_format):
    db = GCDB(tmpdir, format=db_format)

    all_saved_l = []
    for units in ["Mt", "Gt", "Tt"]:
        to_save = create_test_df(
            n_scenarios=10,
            n_variables=1,
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
            units=units,
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


@db_formats
def test_save_overwrite_error(tmpdir, db_format):
    db = GCDB(tmpdir, format=db_format)

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_variables=1,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    dup = cdf(units="m")
    db.save(dup)

    to_save = pix.concat([dup, cdf(units="km")])

    error_msg = re.escape(
        "The following rows are already in the database:\n"
        f"{dup.index.to_frame(index=False)}"
    )
    with pytest.raises(AlreadyInDBError, match=error_msg):
        db.save(to_save)


@db_formats
def test_save_overwrite_force(tmpdir, db_format):
    db = GCDB(Path(tmpdir), format=db_format)

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_variables=1,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    original = cdf(units="m")
    db.save(original)

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(original, loaded)

    original_overwrite = cdf(units="m")
    updated = pix.concat([original_overwrite, cdf(units="km")])

    # With force, we can overwrite
    db.save(updated, allow_overwrite=True)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the new file, but not the original file.
    db_files = list(db.db_dir.glob(f"*.{db_format}"))
    assert set([f.name for f in db_files]) == set(
        f"{prefix}.{db_format}" for prefix in ["1", "index", "filemap"]
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


@db_formats
def test_save_overwrite_force_half_overlap(tmpdir, db_format):
    db = GCDB(Path(tmpdir), format=db_format)

    cdf = partial(
        create_test_df,
        n_variables=5,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="m",
    )

    original = cdf(n_scenarios=6)
    db.save(original)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the file map file plus written file.
    db_files = list(db.db_dir.glob(f"*.{db_format}"))
    assert set([f.name for f in db_files]) == set(
        f"{prefix}.{db_format}" for prefix in ["0", "index", "filemap"]
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
    db_files = list(db.db_dir.glob(f"*.{db_format}"))
    assert set([f.name for f in db_files]) == set(
        f"{prefix}.{db_format}" for prefix in ["1", "2", "index", "filemap"]
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
