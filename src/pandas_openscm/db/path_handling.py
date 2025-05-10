"""
Functionality for handling paths

In order to make our databases portable,
we need to be a bit smarter than just using raw paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import attr
from attrs import define, field


@define
class DBPath:
    """
    Database-related path

    Carries the information required to write paths with certainty
    and keep the database portable.
    """

    abs: Path
    """The absolute path for the file"""

    rel_db: Path = field()
    """The path relative to the database's directory"""

    @rel_db.validator
    def rel_db_validator(self, attribute: attr.Attribute[Any], value: Path) -> None:
        """
        Validate the value of `rel_db`

        Parameters
        ----------
        attribute
            Attribute being set

        value
            Value to use

        Raises
        ------
        AssertionError
            `value` is not within `self.abs`
        """
        if not str(self.abs).endswith(str(value)):
            msg = (
                f"{attribute.name} value, {value!r}, "
                f"is not a sub-path of {self.abs=!r}"
            )
            raise AssertionError(msg)

    @classmethod
    def from_abs_path_and_db_dir(cls, abs: Path, db_dir: Path) -> DBPath:
        """
        Initialise from an absolute path and a database directory

        Parameters
        ----------
        abs
            Absolute path

        db_dir
            Database directory

        Returns
        -------
        :
            Initialised `DBPath`
        """
        return cls(abs=abs, rel_db=abs.relative_to(db_dir))
