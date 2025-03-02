"""
Tests of `pandas_openscm.db.rewriting`
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from pandas_openscm.db import InMemoryDataBackend, InMemoryIndexBackend, OpenSCMDB
from pandas_openscm.db.rewriting import make_move_plan


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
        make_move_plan(
            index_start=pd.Index([1, 2, 3]),
            file_map_start="not used",
            data_to_write="not used",
            get_new_data_file_path=db.get_new_data_file_path,
        )
