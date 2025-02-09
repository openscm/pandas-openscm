"""
Database definition
"""

from __future__ import annotations

import contextlib
import os
from contextlib import nullcontext
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd
from attrs import define

from pandas_openscm.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import pandas_indexing as pix


class OpenSCMDBFormat(StrEnum):
    """Supported database formats"""

    CSV = auto()
    Feather = auto()
    netCDF = auto()
    # Other options to consider:
    #
    # pretty netCDF, where we try and save the data with dimensions
    # where possible
    #
    # HDF5: https://pandas.pydata.org/docs/user_guide/io.html#hdf5-pytables
    # HDF5 = auto()


class AlreadyInDBError(ValueError):
    """
    Raised when saving data would overwrite data which is already in the database
    """

    def __init__(self, already_in_db: pd.DataFrame) -> None:
        """
        Initialise the error

        Parameters
        ----------
        already_in_db
            data that is already in the database
        """
        error_msg = (
            "The following rows are already in the database:\n"
            f"{already_in_db.index.to_frame(index=False)}"
        )
        super().__init__(error_msg)


class EmptyDBError(ValueError):
    """
    Raised when trying to access data from a database that is empty
    """

    def __init__(self, db: OpenSCMDB) -> None:
        """
        Initialise the error

        Parameters
        ----------
        db
            The database
        """
        error_msg = f"The database is empty: {db=}"
        super().__init__(error_msg)


class OpenSCMDBBackend(Protocol):
    """
    Back-end that can be used with [`OpenSCMDB`][(m)]
    """


@define
class OpenSCMDB:
    """
    Database for storing OpenSCM-style data
    """

    backend: OpenSCMDBBackend
    """
    The back-end of the database
    """

    def delete(
        self,
        *,
        progress: bool = False,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> None:
        """
        Delete all data in the database

        Parameters
        ----------
        progress
            Show a progress bar of the deletion's progress

        lock_context_manager
            Context manager to use to acquire the backend's lock file.

            If not supplied, we use
            `self.backend.index_file_lock.acquire`.
        """
        if lock_context_manager is None:
            lock_context_manager = self.backend.index_file_lock.acquire()

        with lock_context_manager:
            to_remove = self.backend.get_all_files()

            if progress:
                try:
                    import tqdm.auto
                except ImportError as exc:
                    raise MissingOptionalDependencyError(  # noqa: TRY003
                        "delete(..., progress=True, ...)", requirement="tqdm"
                    ) from exc

                to_remove = tqdm.auto.tqdm(to_remove, desc="Backend files")

            for f in to_remove:
                os.remove(f)

    def get_new_data_file_path(self, file_id: int) -> Path:
        """
        Get the path in which to write a new data file

        Parameters
        ----------
        file_id
            ID to associate with the file

        Returns
        -------
        :
            File in which to write the new data

        Raises
        ------
        FileExistsError
            A file already exists for the given `file_id`
        """
        file_path = self.backend.get_new_data_file_path(file_id)

        if file_path.exists():
            raise FileExistsError(file_path)

        return file_path

    def load(
        self,
        selector: pd.Index | pd.MultiIndex | pix.selectors.Selector | None = None,
        *,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
        progress: bool = False,
        out_columns_type: type | None = None,
    ) -> pd.DataFrame:
        """
        Load data

        Parameters
        ----------
        selector
            Selector to use to choose the data to load

        lock_context_manager
            Context manager to use to acquire the backend's lock file.

            If not supplied, we use
            `self.backend.index_file_lock.acquire`.

        progress
            Should we use a progress bar to track the loading?

        out_columns_type
            Type to set the output columns to.

            If not supplied, we don't set the output columns' type.

        Returns
        -------
        :
            Loaded data
        """
        if self.backend.is_empty:
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.backend.index_file_lock.acquire()

        def idx_obj(inobj):
            if selector is None:
                res = inobj

            elif isinstance(selector, pd.MultiIndex):
                res = multi_index_lookup(inobj, selector)

            elif isinstance(selector, pd.Index):
                res = inobj[inobj.index.isin(selector.values, level=selector.name)]

            else:
                res = inobj.loc[selector]

            return res

        with lock_context_manager:
            index_raw = self.backend.load_index(lock_context_manager=nullcontext())
            file_map = self.backend.load_file_map(lock_context_manager=nullcontext())

            # Don't need to copy as index_raw is only used internally.
            # The different name is just to help understand the order of operations.
            index = index_raw
            index.index = pd.MultiIndex.from_frame(index_raw)

            index_to_load = idx_obj(index)
            files_to_load = file_map[index_to_load["file_id"].unique()].map(Path)

            if progress:
                try:
                    import tqdm.auto
                except ImportError as exc:
                    raise MissingOptionalDependencyError(  # noqa: TRY003
                        "delete(..., progress=True, ...)", requirement="tqdm"
                    ) from exc

                files_to_load = tqdm.auto.tqdm(files_to_load, desc="Files to load")

            data_l = [self.backend.load_data(f) for f in files_to_load]

        loaded = pix.concat(data_l).set_index(index.index.droplevel("file_id").names)

        # Look up the indexes we want
        # just in case the data we loaded had more than we asked for.
        res = idx_obj(loaded)
        if out_columns_type is not None:
            res.columns = res.columns.astype(out_columns_type)

        return res
