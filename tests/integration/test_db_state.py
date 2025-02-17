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

import hypothesis
import hypothesis.stateful
import numpy as np
import pandas as pd
import pandas_indexing as pix  # noqa: F401
import pytest

from pandas_openscm.db import (
    EmptyDBError,
    OpenSCMDB,
)
from pandas_openscm.testing import (
    assert_frame_alike,
    get_db_data_backends,
    get_db_index_backends,
)

pytestmark = pytest.mark.slow


# @hypothesis.settings(max_examples=50)
@hypothesis.settings(max_examples=1)
class DBMofidier(hypothesis.stateful.RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.dbs = tuple(
            OpenSCMDB(
                backend_data=backend_data(),
                backend_index=backend_index(),
                db_dir=Path(tempfile.mkdtemp()),
            )
            # TODO: split this out so we can do more examples with just memory backend
            # for backend_data in [InMemoryDataBackend]
            # for backend_index in [InMemoryIndexBackend]
            # # # TODO: figure out how to do the type stuff with this.
            # # # Serialising is lossy, so comparisons have to be done carefully.
            for backend_data in get_db_data_backends()
            for backend_index in get_db_index_backends()
        )
        self.data_exp = None
        self.n_ts_options = (1, 3, 5, 10)
        self.timepoint_options = (
            np.arange(2000.0, 2020.0 + 1.0),
            np.arange(2000.0, 2010.0 + 1.0, 2.0),
            np.arange(2010.0, 2020.0 + 1.0, 5.0),
            np.arange(1995.0, 2005.0 + 1.0),
            np.arange(2015.0, 2025.0 + 1.0),
        )
        self.column_options = (
            ("variable", "unit"),
            ("run_id", "variable", "unit"),
            ("scenario", "variable", "unit", "run_id"),
        )
        self.rng = np.random.default_rng()

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
        # TODO: split this out
        n_ts = random.choice(self.n_ts_options)  # noqa: S311
        timepoints = random.choice(self.timepoint_options)  # noqa: S311
        metadata_cols = random.choice(self.column_options)  # noqa: S311
        data_vals = self.rng.random((n_ts, timepoints.size))

        mi_full = []
        n_draws = int(np.ceil(n_ts ** (1 / len(metadata_cols))))
        for col in metadata_cols:
            if self.data_exp is None:
                min_index = 0
            elif col in self.data_exp.index.names:
                min_index = (
                    max(
                        int(v.replace(f"{col}_", "")) if isinstance(v, str) else 0
                        for v in self.data_exp.pix.unique(col)
                    )
                    + 1
                )
            else:
                min_index = 0

            col_vals = [f"{col}_{i}" for i in range(min_index, min_index + n_draws)]
            mi_full.append(col_vals)

        data_index = pd.MultiIndex.from_product(mi_full, names=metadata_cols)
        # Get the number of samples we're interested in
        data_index = data_index[random.sample(range(data_index.shape[0]), n_ts)]

        data = pd.DataFrame(data_vals, index=data_index, columns=timepoints)

        for db in self.dbs:
            # Should be no overlap hence no overwrite needed
            db.save(data)

        if self.data_exp is None:
            self.data_exp = data
        else:
            self.data_exp = pd.concat(
                v.dropna(axis="rows", how="all")
                for v in self.data_exp.align(data, axis="rows")
            )

    # TODO:
    # - Add fully overlapping data
    # - Add partially overlapping data
    # - Add grouped dated

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


DBModifierTest = DBMofidier.TestCase
