"""
Basic unit tests of `pandas_openscm.database`
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import (
    EmptyDBError,
    InMemoryDataBackend,
    InMemoryIndexBackend,
    OpenSCMDB,
    SaveAction,
    save_file,
)
from pandas_openscm.testing import create_test_df


def test_get_existing_data_file_path(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    fp = db.get_new_data_file_path(file_id=10)

    # Assume the file gets written somewhere
    fp.touch()

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
    )

    with expecation:
        getattr(db, meth)(*args)


def test_make_move_plan_index_start_not_multi_error(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
    )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "`index_start` must be an instance of `pd.MultiIndex`. "
            "Received type(index_start)=<class 'pandas.core.indexes.base.Index'>"
        ),
    ):
        db.make_move_plan(
            index_start=pd.Index([1, 2, 3]),
            file_map_start="not used",
            data_to_write="not used",
        )


def test_save_data_index_not_multi_error(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
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


def test_save_file_unrecognised_save_action_info_kind_error():
    info_kind = 12

    with pytest.raises(NotImplementedError, match=str(info_kind)):
        save_file(
            SaveAction(
                info="not used",
                info_kind=info_kind,
                backend="not used",
                save_path="not used",
            )
        )
