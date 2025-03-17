"""
Basic unit tests of `pandas_openscm.testing`
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.testing import (
    get_parametrized_db_data_backends,
    get_parametrized_db_index_backends,
)


@pytest.mark.parametrize(
    "to_call, exp_name, args",
    (
        (get_parametrized_db_data_backends, "get_parametrized_db_data_backends", []),
        (get_parametrized_db_index_backends, "get_parametrized_db_index_backends", []),
    ),
)
def test_tqdm_not_available(to_call, exp_name, args):
    with patch.dict(sys.modules, {"pytest": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=(f"`{exp_name}` requires pytest to be installed"),
        ):
            to_call(*args)
