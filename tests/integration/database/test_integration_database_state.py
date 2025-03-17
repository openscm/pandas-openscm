"""
Stateful testing of the database with [hypothesis](https://hypothesis.readthedocs.io/en/latest/)

This allows us check that a series of operations on the database
yields the same result, independent of data and index back-ends.
It's not a perfect test, but it is a very helpful one for finding edge cases
and making sure that different combinations of operations all work.
"""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import (
    EmptyDBError,
    InMemoryDataBackend,
    InMemoryIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.testing import (
    assert_frame_alike,
    get_db_data_backends,
    get_db_index_backends,
)

pytest.importorskip("filelock")
hypothesis = pytest.importorskip("hypothesis")
hypothesis_stateful = pytest.importorskip("hypothesis.stateful")
pytest.importorskip("pandas_indexing")
pytestmark = pytest.mark.slow


def get_new_data(
    *,
    n_ts_options: tuple[int, ...],
    timepoint_options: tuple[np.typing.NDArray[float], ...],
    metadata_options: tuple[tuple[str, ...]],
    rng: np.random.Generator,
    data_existing: pd.DataFrame | None,
) -> pd.DataFrame:
    n_ts = random.choice(n_ts_options)  # noqa: S311
    timepoints = random.choice(timepoint_options)  # noqa: S311
    metadata_cols = random.choice(metadata_options)  # noqa: S311

    data_vals = rng.random((n_ts, timepoints.size))
    multi_index_full = []
    n_draws = int(np.ceil(n_ts ** (1 / len(metadata_cols))))
    for col in metadata_cols:
        if data_existing is None:
            min_index = 0
        elif col in data_existing.index.names:
            min_index = (
                max(
                    int(v.replace(f"{col}_", "")) if isinstance(v, str) else 0
                    for v in data_existing.pix.unique(col)
                )
                + 1
            )
        else:
            min_index = 0

        col_vals = [f"{col}_{i}" for i in range(min_index, min_index + n_draws)]
        multi_index_full.append(col_vals)

    data_index = pd.MultiIndex.from_product(multi_index_full, names=metadata_cols)
    # Get the number of samples we're interested in
    data_index = data_index[random.sample(range(data_index.shape[0]), n_ts)]

    new_data = pd.DataFrame(data_vals, index=data_index, columns=timepoints)

    return new_data


class DBMofidierBase(hypothesis.stateful.RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.data_exp = None
        self.rng = np.random.default_rng()
        for db in self.dbs:
            db.db_dir.mkdir(exist_ok=True, parents=True)

    def teardown(self):
        for db in self.dbs:
            shutil.rmtree(db.db_dir)

    @hypothesis.stateful.rule()
    def delete(self):
        self.data_exp = None
        for db in self.dbs:
            # Should work irrespective of whether
            # there is anything to delete or not
            db.delete()

    @hypothesis.stateful.rule()
    def add_new_data(self):
        new_data = get_new_data(
            n_ts_options=self.n_ts_options,
            timepoint_options=self.timepoint_options,
            metadata_options=self.metadata_options,
            rng=self.rng,
            data_existing=self.data_exp,
        )

        for db in self.dbs:
            # Should be no overlap hence no overwrite needed
            db.save(new_data)

        if self.data_exp is None:
            self.data_exp = new_data
        else:
            self.data_exp = pd.concat(
                v.dropna(axis="rows", how="all")
                for v in self.data_exp.align(new_data, axis="rows")
            )

    @hypothesis.stateful.rule()
    def add_new_data_grouped(self):
        metadata_options = self.metadata_options

        new_data = get_new_data(
            n_ts_options=self.n_ts_options,
            timepoint_options=self.timepoint_options,
            metadata_options=metadata_options,
            rng=self.rng,
            data_existing=self.data_exp,
        )

        groupby = random.sample(new_data.index.names, len(new_data.index.names) - 1)
        for db in self.dbs:
            # Should be no overlap hence no overwrite needed
            db.save(new_data, groupby=groupby)

        if self.data_exp is None:
            self.data_exp = new_data
        else:
            self.data_exp = pd.concat(
                v.dropna(axis="rows", how="all")
                for v in self.data_exp.align(new_data, axis="rows")
            )

    @hypothesis.stateful.precondition(lambda self: self.data_exp is not None)
    @hypothesis.stateful.rule()
    def add_fully_overlapping_data(self):
        existing_idx = self.dbs[0].load_index()
        to_dup_idx = existing_idx[
            existing_idx["file_id"] == existing_idx["file_id"].min()
        ].index

        to_dup_loc = self.data_exp.index.isin(to_dup_idx)
        dup = self.data_exp.loc[to_dup_loc, :] * 1.1

        for db in self.dbs:
            # Should not need warn_on_partial_overwrite
            # here as we're doing a complete overwrite.
            db.save(dup, allow_overwrite=True)

        self.data_exp = pd.concat(
            [
                self.data_exp.loc[~to_dup_loc, :],
                dup,
            ]
        )

    @hypothesis.stateful.precondition(lambda self: self.data_exp is not None)
    @hypothesis.stateful.rule()
    def add_partially_overlapping_data(self):
        existing_idx = self.dbs[0].load_index()

        file_ids_unique = existing_idx["file_id"].unique()
        n_file_ids = len(file_ids_unique)
        to_dup_idx = None
        for i in range(min(3, n_file_ids)):
            tmp = existing_idx[existing_idx["file_id"] == file_ids_unique[i]].index
            tmp = tmp[: min(3, tmp.size)]
            if to_dup_idx is None:
                to_dup_idx = tmp
            else:
                to_dup_idx = to_dup_idx.append(tmp)

        to_dup_loc = self.data_exp.index.isin(to_dup_idx)
        dup = self.data_exp.loc[to_dup_loc, :] * 1.1

        for db in self.dbs:
            db.save(dup, allow_overwrite=True, warn_on_partial_overwrite=False)

        self.data_exp = pd.concat(
            [
                self.data_exp.loc[~to_dup_loc, :],
                dup,
            ]
        )

    @hypothesis.stateful.invariant()
    def all_db_index_are_multiindex(self):
        if self.data_exp is None:
            return

        assert isinstance(self.data_exp.index, pd.MultiIndex)
        for db in self.dbs:
            try:
                index = db.load_index()
                assert isinstance(index.index, pd.MultiIndex)
            except AssertionError as exc:
                msg = (
                    f"{type(db.backend_data).__name__=}"
                    f"{type(db.backend_index).__name__=}"
                )
                raise AssertionError(msg) from exc

    @hypothesis.stateful.invariant()
    def all_dbs_consistent_with_expected(self):
        for db in self.dbs:
            try:
                if self.data_exp is not None:
                    loaded = db.load(out_columns_type=self.data_exp.columns.dtype)
                    assert isinstance(loaded.index, pd.MultiIndex)
                    assert_frame_alike(loaded, self.data_exp)

                    loaded_metadata = db.load_metadata()
                    assert isinstance(loaded_metadata, pd.MultiIndex)
                    loaded_comparison = (
                        loaded_metadata.to_frame(index=False)
                        .fillna("i_was_nan")
                        .replace("nan", "i_was_nan")
                        .sort_values(self.data_exp.index.names)
                        .reset_index(drop=True)
                    )
                    exp_comparison = (
                        self.data_exp.index.to_frame(index=False)
                        .fillna("i_was_nan")
                        .sort_values(self.data_exp.index.names)
                        .reset_index(drop=True)
                    )
                    pd.testing.assert_frame_equal(
                        loaded_comparison, exp_comparison, check_like=True
                    )

                else:
                    with pytest.raises(EmptyDBError):
                        db.load()

            except AssertionError as exc:
                msg = (
                    f"{type(db.backend_data).__name__=}"
                    f"{type(db.backend_index).__name__=}"
                )
                raise AssertionError(msg) from exc


@hypothesis.settings(max_examples=5)
class DBMofidier(DBMofidierBase):
    dbs = tuple(
        OpenSCMDB(
            backend_data=backend_data(),
            backend_index=backend_index(),
            db_dir=Path(tempfile.mkdtemp()),
        )
        for backend_data in get_db_data_backends()
        for backend_index in get_db_index_backends()
    )
    n_ts_options = (1, 3)
    timepoint_options = (
        np.arange(1995.0, 2005.0 + 1.0),
        np.arange(2000.0, 2020.0 + 1.0),
        np.arange(2015.0, 2025.0 + 1.0),
    )
    metadata_options = (
        ("variable", "unit"),
        ("run_id", "variable", "unit"),
    )


@hypothesis.settings(max_examples=20)
class DBMofidierInMemory(DBMofidierBase):
    dbs = tuple(
        OpenSCMDB(
            backend_data=backend_data(),
            backend_index=backend_index(),
            db_dir=Path(tempfile.mkdtemp()),
        )
        for backend_data in [InMemoryDataBackend]
        for backend_index in [InMemoryIndexBackend]
    )
    n_ts_options = (1, 3, 5, 10)
    timepoint_options = (
        np.arange(2000.0, 2020.0 + 1.0),
        np.arange(2000.0, 2010.0 + 1.0, 2.0),
        np.arange(2010.0, 2020.0 + 1.0, 5.0),
        np.arange(1995.0, 2005.0 + 1.0),
        np.arange(2015.0, 2025.0 + 1.0),
    )
    metadata_options = (
        ("variable", "unit"),
        ("run_id", "variable", "unit"),
        ("scenario", "variable", "unit", "run_id"),
    )


DBModifierTest = pytest.mark.superslow(DBMofidier.TestCase)
DBModifierInMemoryTest = DBMofidierInMemory.TestCase
