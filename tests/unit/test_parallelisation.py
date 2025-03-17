"""
Basic unit tests of `pandas_openscm.parallelisation`
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.parallelisation import get_tqdm_auto


@pytest.mark.parametrize(
    "to_call, exp_name, args",
    ((get_tqdm_auto, "get_tqdm_auto", []),),
)
def test_tqdm_not_available(to_call, exp_name, args):
    with patch.dict(sys.modules, {"tqdm": None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=(f"`{exp_name}` requires tqdm to be installed"),
        ):
            to_call(*args)
