"""
Basic unit tests of `pandas_openscm.database.netcdf`
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pandas as pd
import pytest

from pandas_openscm.db import netCDFDataBackend, netCDFIndexBackend
from pandas_openscm.db.netcdf import metadata_df_to_xr
from pandas_openscm.exceptions import MissingOptionalDependencyError


@pytest.mark.parametrize(
    "to_call, exp_name, args",
    (
        (netCDFDataBackend().load_data, "netCDFBackend.load_data", ["file_path"]),
        (
            netCDFDataBackend().save_data,
            "netCDFBackend.save_data",
            ["data", "file_path"],
        ),
        (
            netCDFIndexBackend().load_file_map,
            "netCDFBackend.load_file_map",
            ["file_path"],
        ),
        (
            netCDFIndexBackend().load_index,
            "netCDFBackend.load_index",
            ["file_path"],
        ),
        (
            netCDFIndexBackend().save_file_map,
            "netCDFBackend.save_file_map",
            ["file_map", "file_path"],
        ),
        (
            netCDFIndexBackend().save_index,
            "metadata_df_to_xr",
            [pd.DataFrame(), "file_path"],
        ),
        (
            metadata_df_to_xr,
            "metadata_df_to_xr",
            ["metadata"],
        ),
    ),
)
def test_xarray_not_available(to_call, exp_name, args):
    with patch.dict(sys.modules, {"xarray": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=(f"`{exp_name}` requires xarray to be installed"),
        ):
            to_call(*args)
