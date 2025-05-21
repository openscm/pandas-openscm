"""
Tests of moving the database
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import (
    CSVDataBackend,
    CSVIndexBackend,
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
    netCDFDataBackend,
    netCDFIndexBackend,
)
from pandas_openscm.testing import assert_frame_alike

pytest.importorskip("filelock")


@pytest.mark.parametrize(
    "backend_data, backend_index",
    (
        pytest.param(
            FeatherDataBackend(),
            FeatherIndexBackend(),
            id="feather",
        ),
        pytest.param(
            netCDFDataBackend(),
            netCDFIndexBackend(),
            id="netCDF",
        ),
        pytest.param(
            CSVDataBackend(),
            CSVIndexBackend(),
            id="csv",
        ),
    ),
)
@pytest.mark.parametrize("provide_backend_data_to_class_method", (True, False))
@pytest.mark.parametrize("provide_backend_index_to_class_method", (True, False))
def test_move_db(  # noqa: PLR0913
    provide_backend_index_to_class_method,
    provide_backend_data_to_class_method,
    backend_data,
    backend_index,
    tmpdir,
    setup_pandas_accessor,
):
    initial_db_dir = Path(tmpdir) / "initial"
    other_db_dir = Path(tmpdir) / "other"
    tar_archive = Path(tmpdir) / "tar_archive.tar.gz"

    db = OpenSCMDB(
        db_dir=initial_db_dir,
        backend_data=backend_data,
        backend_index=backend_index,
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
    from_gzipped_tar_archive_kwargs = {}
    if provide_backend_data_to_class_method:
        from_gzipped_tar_archive_kwargs["backend_data"] = backend_data

    if provide_backend_index_to_class_method:
        from_gzipped_tar_archive_kwargs["backend_index"] = backend_index

    db_other = OpenSCMDB.from_gzipped_tar_archive(
        tar_archive, db_dir=other_db_dir, **from_gzipped_tar_archive_kwargs
    )

    # Delete the original
    db.delete()

    assert_frame_alike(df_timeseries_like, db_other.load(out_columns_type=int))

    locator = pd.Index(["scenario_b"], name="scenario")
    assert_frame_alike(
        df_timeseries_like.openscm.mi_loc(locator),
        db_other.load(locator, out_columns_type=int),
    )
