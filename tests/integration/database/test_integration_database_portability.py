"""
Tests of moving the database
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import CSVDataBackend, CSVIndexBackend, OpenSCMDB
from pandas_openscm.testing import assert_frame_alike


@pytest.mark.parametrize(
    "backend_data_for_class_method, backend_index_for_class_method",
    (
        pytest.param(
            CSVDataBackend(),
            CSVIndexBackend(),
            id="provided",
        ),
        pytest.param(
            None,
            None,
            id="guessed",
        ),
    ),
)
def test_move_db(
    backend_data_for_class_method,
    backend_index_for_class_method,
    tmpdir,
    setup_pandas_accessor,
):
    initial_db_dir = Path(tmpdir) / "initial"
    other_db_dir = Path(tmpdir) / "other"
    tar_archive = Path(tmpdir) / "tar_archive.tar.gz"

    db = OpenSCMDB(
        db_dir=initial_db_dir,
        backend_data=CSVDataBackend(),
        backend_index=CSVIndexBackend(),
    )

    df_timeseries_like = pd.DataFrame(
        np.arange(12).reshape(4, 3),
        columns=[2010, 2015, 2025],
        index=pd.MultiIndex.from_tuples(
            [
                ("scenario_a", "climate_model_a", "Temperature", "K"),
                ("scenario_b", "climate_model_a", "Temperature", "K"),
                ("scenario_b", "climate_model_b", "Temperature", "K"),
                ("scenario_b", "climate_model_b", "Ocean Heat Uptake", "J"),
            ],
            names=["scenario", "climate_model", "variable", "unit"],
        ),
    )

    db.save(df_timeseries_like, groupby=["scenario", "variable"])

    # Create a tar archive (returns the archive path, even though it's also an input)
    tar_archive = db.to_gzipped_tar_archive(tar_archive)

    # Expand elsewhere
    db_other = OpenSCMDB.from_gzipped_tar_archive(
        tar_archive,
        db_dir=other_db_dir,
        backend_data=backend_data_for_class_method,
        backend_index=backend_index_for_class_method,
    )

    # Delete the original
    db.delete()

    assert_frame_alike(df_timeseries_like, db_other.load())

    locator = pd.Index(["scenario_b"], name="scenario")
    assert_frame_alike(
        df_timeseries_like.openscm.mi_loc(locator), db_other.load(locator)
    )
