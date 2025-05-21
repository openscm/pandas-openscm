"""
Database
"""

from __future__ import annotations

from pandas_openscm.db.backends import DATA_BACKENDS, INDEX_BACKENDS
from pandas_openscm.db.csv import CSVDataBackend, CSVIndexBackend
from pandas_openscm.db.feather import FeatherDataBackend, FeatherIndexBackend
from pandas_openscm.db.in_memory import InMemoryDataBackend, InMemoryIndexBackend
from pandas_openscm.db.interfaces import OpenSCMDBDataBackend, OpenSCMDBIndexBackend
from pandas_openscm.db.netcdf import netCDFDataBackend, netCDFIndexBackend
from pandas_openscm.db.openscm_db import AlreadyInDBError, EmptyDBError, OpenSCMDB

__all__ = [
    "DATA_BACKENDS",
    "INDEX_BACKENDS",
    "AlreadyInDBError",
    "CSVDataBackend",
    "CSVIndexBackend",
    "EmptyDBError",
    "FeatherDataBackend",
    "FeatherIndexBackend",
    "InMemoryDataBackend",
    "InMemoryIndexBackend",
    "OpenSCMDB",
    "OpenSCMDBDataBackend",
    "OpenSCMDBIndexBackend",
    "netCDFDataBackend",
    "netCDFIndexBackend",
]
