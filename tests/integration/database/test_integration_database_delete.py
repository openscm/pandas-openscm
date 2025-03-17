"""
Tests of deleting with `pandas_openscm.OpenSCMDB`
"""

from __future__ import annotations

import contextlib
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
    create_test_df,
)


def test_delete(tmpdir):
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=InMemoryDataBackend(),
        backend_index=InMemoryIndexBackend(),
        index_file_lock=contextlib.nullcontext(),  # not used
    )

    db.save(
        create_test_df(
            n_scenarios=10,
            variables=[("variable_1", "kg"), ("variable_2", "Mt"), ("variable_3", "m")],
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        )
    )

    assert isinstance(db.load(), pd.DataFrame)

    db.delete()

    with pytest.raises(EmptyDBError):
        db.load_metadata()

    with pytest.raises(EmptyDBError):
        db.load()
