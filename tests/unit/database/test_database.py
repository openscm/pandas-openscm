"""
Basic unit tests of `pandas_openscm.database`
"""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import (
    DATA_BACKENDS,
    INDEX_BACKENDS,
    CSVDataBackend,
    CSVIndexBackend,
    EmptyDBError,
    FeatherDataBackend,
    FeatherIndexBackend,
    InMemoryDataBackend,
    InMemoryIndexBackend,
    OpenSCMDB,
    netCDFDataBackend,
    netCDFIndexBackend,
)
from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.testing import create_test_df


def test_available_backends_data():
    assert isinstance(DATA_BACKENDS.get_instance("csv"), CSVDataBackend)
    assert isinstance(DATA_BACKENDS.get_instance("feather"), FeatherDataBackend)
    assert isinstance(DATA_BACKENDS.get_instance("in_memory"), InMemoryDataBackend)
    assert isinstance(DATA_BACKENDS.get_instance("netCDF"), netCDFDataBackend)


def test_unavailable_data_backend():
    with pytest.raises(
        KeyError, match=re.escape("option='junk' is not supported. Available options:")
    ):
        DATA_BACKENDS.get_instance("junk")


def test_guess_backend_data():
    assert isinstance(DATA_BACKENDS.guess_backend("0.csv"), CSVDataBackend)
    assert isinstance(DATA_BACKENDS.guess_backend("0.feather"), FeatherDataBackend)
    assert isinstance(DATA_BACKENDS.guess_backend("0.in-mem"), InMemoryDataBackend)
    assert isinstance(DATA_BACKENDS.guess_backend("0.nc"), netCDFDataBackend)


def test_guess_data_backend_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Could not guess backend from data_file_name='0.junk'. "
            "The file's extension does not match any of the available options: "
            "known_options_and_extensions="
        ),
    ):
        DATA_BACKENDS.guess_backend("0.junk")


def test_available_backends_index():
    assert isinstance(INDEX_BACKENDS.get_instance("csv"), CSVIndexBackend)
    assert isinstance(INDEX_BACKENDS.get_instance("feather"), FeatherIndexBackend)
    assert isinstance(INDEX_BACKENDS.get_instance("in_memory"), InMemoryIndexBackend)
    assert isinstance(INDEX_BACKENDS.get_instance("netCDF"), netCDFIndexBackend)


def test_unavailable_index_backend():
    with pytest.raises(
        KeyError, match=re.escape("option='junk' is not supported. Available options:")
    ):
        INDEX_BACKENDS.get_instance("junk")


def test_guess_backend_index():
    assert isinstance(INDEX_BACKENDS.guess_backend("0.csv"), CSVIndexBackend)
    assert isinstance(INDEX_BACKENDS.guess_backend("0.feather"), FeatherIndexBackend)
    assert isinstance(INDEX_BACKENDS.guess_backend("0.in-mem"), InMemoryIndexBackend)
    assert isinstance(INDEX_BACKENDS.guess_backend("0.nc"), netCDFIndexBackend)


def test_guess_index_backend_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Could not guess backend from index_file_name='index.junk'. "
            "The file's extension does not match any of the available options: "
            "known_options_and_extensions="
        ),
    ):
        INDEX_BACKENDS.guess_backend("index.junk")


def test_filelock_not_available_default_initialisation(tmpdir):
    with patch.dict(sys.modules, {"filelock": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=("`default_index_file_lock` requires filelock to be installed"),
        ):
            OpenSCMDB(
                db_dir=Path(tmpdir),
                backend_data=InMemoryDataBackend(),
                backend_index=InMemoryIndexBackend(),
            )

        # Not an issue if we bypass the lock
        OpenSCMDB(
            db_dir=Path(tmpdir),
            backend_data=InMemoryDataBackend(),
            backend_index=InMemoryIndexBackend(),
            index_file_lock=nullcontext(),
        )


def test_get_existing_data_file_path(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
    )

    fp = db.get_new_data_file_path(file_id=10)

    # Assume the file gets written somewhere
    fp.abs.touch()

    with pytest.raises(FileExistsError):
        db.get_new_data_file_path(file_id=10)


@pytest.mark.parametrize(
    "meth, args, expecation",
    (
        *[
            (meth, args, nullcontext())
            for meth, args in [
                ("delete", []),
                ("get_new_data_file_path", [0]),
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
            ]
        ],
        *[
            (
                meth,
                args,
                pytest.raises(EmptyDBError, match="The database is empty: db="),
            )
            for meth, args in [
                ("load", []),
                ("load_file_map", []),
                ("load_index", []),
                ("load_metadata", []),
            ]
        ],
    ),
)
def test_raise_if_empty(tmpdir, meth, args, expecation):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
    )

    with expecation:
        getattr(db, meth)(*args)


def test_save_data_index_not_multi_error(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
    )

    data = pd.DataFrame([0, 1], index=pd.Index(["a", "b"]))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "`data.index` must be an instance of `pd.MultiIndex`. "
            "Received type(data.index)=<class 'pandas.core.indexes.base.Index'>"
        ),
    ):
        db.save(data)


def test_save_data_duplicate_index_rows(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),  # not used
    )

    data = pd.DataFrame(
        [
            [1.0, 2.0],
            [8.0, 1.0],
            [9.0, 7.0],
            [3.0, 5.0],
        ],
        columns=[2010.0, 2020.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("scen_a", "mod_a", "var_a"),
                ("scen_b", "mod_b", "var_b"),
                ("scen_a", "mod_a", "var_a"),
                ("scen_a", "mod_b", "var_b"),
            ],
            names=["scenario", "model", "variable"],
        ),
    )

    duplicates = data.loc[data.index.duplicated(keep=False)]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"`data` contains rows with the same metadata. duplicates=\n{duplicates}"
        ),
    ):
        db.save(data)


def test_filelock_not_available_default_reader_lock(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=nullcontext(),
    )
    # Need some data in there so the reader can be created
    data = pd.DataFrame(
        [[1.0, 2.0]],
        columns=[2010.0, 2020.0],
        index=pd.MultiIndex.from_tuples(
            [("scen_a", "mod_a", "var_a")],
            names=["scenario", "model", "variable"],
        ),
    )
    db.save(data)

    with patch.dict(sys.modules, {"filelock": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=re.escape(
                "`create_reader(..., lock=True, ...)` requires filelock to be installed"
            ),
        ):
            db.create_reader()

        # If we disable lock creation, all fine
        db.create_reader(lock=False)

        # Or pass in our own lock, also all fine
        db.create_reader(lock=nullcontext())
