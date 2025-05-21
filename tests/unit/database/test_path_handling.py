"""
Tests of `pandas_openscm.db.path_handling`
"""

import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from pandas_openscm.db.path_handling import DBPath


@pytest.mark.parametrize(
    "abs, rel_db, exp",
    (
        (Path("/a/b/c/d.csv"), Path("d.csv"), does_not_raise()),
        (Path("/a/b/c/d.csv"), Path("c/d.csv"), does_not_raise()),
        (
            Path("/a/b/c/d.csv"),
            Path("e/d.csv"),
            pytest.raises(
                AssertionError,
                match="".join(
                    (
                        re.escape("rel_db value, "),
                        ".*Path",
                        re.escape(r"('e/d.csv'), is not a sub-path of self.abs="),
                        ".*Path",
                        re.escape("('/a/b/c/d.csv')"),
                    )
                ),
            ),
        ),
    ),
)
def test_rel_db_validator(abs, rel_db, exp):
    with exp:
        DBPath(abs=abs, rel_db=rel_db)
