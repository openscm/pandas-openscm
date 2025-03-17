"""
Tests of speed of `pandas_openscm.OpenSCMDB`

These tests aim to catch places
where serialising data via OpenSCMDB
is much slower than directly using pandas.
These aren't perfect, but they're better than zero
for catching obvious mistakes.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.testing import create_test_df


@pytest.mark.parametrize(
    "groupby",
    (
        pytest.param(None, id="single-file"),
        pytest.param(["scenario"], id="scenario-groups"),
        pytest.param(["variable", "scenario"], id="scenario-variable-groups"),
    ),
)
def test_overhead(groupby, tmpdir):
    # Want lock time included in this test,
    # hence skip if not available
    pytest.importorskip("filelock")

    tmpdir = Path(tmpdir)

    feather_data_be = FeatherDataBackend()
    feather_index_be = FeatherIndexBackend()

    db = OpenSCMDB(
        db_dir=tmpdir / "db",
        backend_data=feather_data_be,
        backend_index=feather_index_be,
    )

    df = create_test_df(
        variables=[("Temperature", "K")],
        n_scenarios=150,
        n_runs=600,
        timepoints=np.arange(200.0),
    )

    pandas_out_dir = tmpdir / "pandas"
    pandas_out_dir.mkdir(parents=True)

    start_pandas_save = time.perf_counter()

    # Semi-replicate what the db has to do
    index = pd.DataFrame(
        np.full(df.shape[0], "path-to-somewhere"), index=df.index, columns=["file_path"]
    )
    index.to_feather(pandas_out_dir / "index.feather")

    if groupby is None:
        feather_data_be.save_data(df, pandas_out_dir / "data-pandas.feather")
    else:
        for i, (_, gdf) in enumerate(df.groupby(groupby)):
            gdf.to_feather(pandas_out_dir / f"{i}_data-pandas.feather")

    stop_pandas_save = time.perf_counter()
    time_pandas_save = stop_pandas_save - start_pandas_save

    start_db_save = time.perf_counter()

    db.db_dir.mkdir()
    db.save(df, groupby=groupby)

    stop_db_save = time.perf_counter()
    time_db_save = stop_db_save - start_db_save

    # Ok, particularly given how few files we're dealing with.
    # This is mainly about avoiding a factor of 10
    # (which was the difference we were getting in earlier implemenations).
    tol_save = 0.25
    tol_load = 0.2

    overhead = (time_db_save - time_pandas_save) / time_pandas_save
    assert overhead <= tol_save, f"Overhead is more than {tol_save*100}%"

    start_pandas_load = time.perf_counter()

    [pd.read_feather(f) for f in pandas_out_dir.glob("*.feather")]

    stop_pandas_load = time.perf_counter()
    time_pandas_load = stop_pandas_load - start_pandas_load

    start_db_load = time.perf_counter()

    db.load()

    stop_db_load = time.perf_counter()
    time_db_load = stop_db_load - start_db_load

    overhead = (time_db_load - time_pandas_load) / time_pandas_load
    assert overhead <= tol_load, f"Overhead is more than {tol_load*100}%"
