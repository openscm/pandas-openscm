"""
Pandas accessors for OpenSCM-related functionality.
"""

import importlib.metadata

from pandas_openscm.accessors import register_pandas_accessor

__version__ = importlib.metadata.version("pandas_openscm")

__all__ = ["register_pandas_accessor"]
