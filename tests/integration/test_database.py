"""
Tests of `pandas_openscm.OpenSCMDB`
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import filelock
import numpy as np
import pytest

from pandas_openscm.testing import create_test_df, get_parametrized_db_formats

db_formats = get_parametrized_db_formats()


@pytest.mark.parametrize(
    "meth, args",
    (
        ("delete", []),
        ("load", []),
        ("load_metadata", []),
        ("regroup", [["scenarios"]]),
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
@db_formats
def test_locking(tmpdir, meth, args, db_format):
    db = GCDB(Path(tmpdir), format=db_format)

    # Put some data in the db so there's something to lock
    db.save(
        create_test_df(
            n_scenarios=1,
            n_variables=1,
            n_runs=1,
            timepoints=np.array([10.0]),
            units="Mt",
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


@db_formats
def test_load_with_loc(tmpdir, db_format):
    db = GCDB(tmpdir, format=db_format)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

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

        pd.testing.assert_frame_equal(loaded, exp)


@db_formats
def test_load_with_index_all(tmpdir, db_format):
    db = GCDB(tmpdir, format=db_format)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

    idx = full_db.index
    exp = full_db

    loaded = db.load(idx, out_columns_type=float)

    pd.testing.assert_frame_equal(loaded, exp)


@pytest.mark.parametrize(
    "slice",
    (slice(None, None, None), slice(None, 3, None), slice(2, 4, None), slice(1, 15, 2)),
)
@db_formats
def test_load_with_index_slice(tmpdir, slice, db_format):
    db = GCDB(tmpdir, format=db_format)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

    idx = full_db.index[slice]
    exp = full_db[slice]

    loaded = db.load(idx, out_columns_type=float)

    pd.testing.assert_frame_equal(loaded, exp)


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
@db_formats
def test_load_with_pix_unique_levels(tmpdir, levels, db_format):
    db = GCDB(tmpdir, format=db_format)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

    locator = None
    for level in levels:
        if locator is None:
            locator = pix.isin(**{level: full_db.pix.unique(level)[:2]})
        else:
            locator &= pix.isin(**{level: full_db.pix.unique(level)[:2]})

    exp = full_db.loc[locator]
    idx = exp.pix.unique(levels)

    loaded = db.load(idx, out_columns_type=float)

    pd.testing.assert_frame_equal(loaded, exp)


@db_formats
def test_deletion(tmpdir, db_format):
    db = GCDB(Path(tmpdir), format=db_format)

    db.save(
        create_test_df(
            n_scenarios=10,
            n_variables=3,
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
            units="Mt",
        )
    )

    assert isinstance(db.load(), pd.DataFrame)

    db.delete()

    with pytest.raises(EmptyDBError):
        db.load_metadata()

    with pytest.raises(EmptyDBError):
        db.load()


@db_formats
def test_regroup(tmpdir, db_format):
    db = GCDB(Path(tmpdir), format=db_format)

    all_dat = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    db.save(all_dat)

    pd.testing.assert_frame_equal(db.load(out_columns_type=float), all_dat)
    # Testing implementation but ok as a helper for now
    assert len(list(db.db_dir.glob(f"*.{db_format}"))) == 3

    for new_grouping in (
        ["scenario"],
        ["scenario", "variable"],
        ["variable", "run"],
    ):
        db.regroup(new_grouping, progress=True)

        # Make sure data unchanged
        pd.testing.assert_frame_equal(
            db.load(out_columns_type=float), all_dat, check_like=True
        )
        # Testing implementation but ok as a helper for now
        assert (
            len(list(db.db_dir.glob(f"*.{db_format}")))
            == 2 + all_dat.pix.unique(new_grouping).shape[0]
        )
