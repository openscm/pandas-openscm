"""
Database definition
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import filelock
import pandas as pd
from attrs import define

from pandas_openscm.db.csv import CSVBackend
from pandas_openscm.db.feather import FeatherBackend
from pandas_openscm.db.netcdf import netCDFBackend
from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.pandas_helpers import multi_index_lookup, multi_index_match
from pandas_openscm.parallelisation import (
    ProgressLike,
    apply_op_parallel_progress,
    figure_out_progress_bars,
)

if TYPE_CHECKING:
    import pandas_indexing as pix  # type: ignore # see https://github.com/coroa/pandas-indexing/pull/63
    import tqdm.asyncio


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
    Backend that can be used with [`OpenSCMDB`][(m)]
    """

    ext: str
    """
    Extension to use with files saved by this backend.
    """

    @staticmethod
    def load_data_file(data_file: Path) -> pd.DataFrame:
        """
        Load a data file

        This is a low-level method
        that just handles the specifics of serialising the data to disk.
        Working out the path in which to save the data
        should happen in higher-level functions.

        Parameters
        ----------
        data_file
            File from which to load the data

        Returns
        -------
        :
            Loaded data
        """

    @staticmethod
    def load_file_map(file_map_file: Path) -> pd.DataFrame:
        """
        Load the database's file map

        This is a low-level method
        that just handles the specifics of serialising the index to disk.
        Working out the path in which to save the file map
        should happen in higher-level functions.

        Parameters
        ----------
        file_map_file
            File from which to load the file map

        Returns
        -------
        :
            Loaded file map
        """

    @staticmethod
    def load_index(index_file: Path) -> pd.DataFrame:
        """
        Load the database's index

        This is a low-level method
        that just handles the specifics of serialising the index to disk.
        Working out the path in which to save the index
        should happen in higher-level functions.

        Parameters
        ----------
        index_file
            File from which to load the index

        Returns
        -------
        :
            Loaded index
        """

    @staticmethod
    def save_data(
        data: pd.DataFrame,
        data_file: Path,
    ) -> None:
        """
        Save an individual data file to the database

        This is a low-level method
        that just handles the specifics of serialising the data to disk.

        Parameters
        ----------
        data
            Data to save

        data_file
            File in which to save the data
        """

    def save_database(  # noqa: PLR0913
        self,
        index: pd.DataFrame,
        index_file: Path,
        file_map: pd.Series[Path],  # type: ignore # pandas confused about what it supports
        file_map_file: Path,
        data: pd.DataFrame,
        data_file: Path,
    ) -> None:
        """
        Save the database

        This is a low-level method
        that just handles the specifics of serialising
        the database components to the disk.
        Working out what data to save and in what path
        should happen in higher-level functions.

        Parameters
        ----------
        index
            Index file to save

        index_file
            File in which to save the index

        file_map
            File map to save

        file_map_file
            File in which to save the file map

        data
            Data to save

        data_file
            File in which to save the data
        """


@define
class OpenSCMDB:
    """
    Database for storing OpenSCM-style data

    This class is focussed on backends that use files as their storage.
    If you had a different database backend,
    you might make different choices.
    We haven't thought through those use cases
    hence aren't sure how much effort
    would be required to make something truly backend agnostic.
    """

    backend: OpenSCMDBBackend
    """
    The backend of the database
    """

    db_dir: Path
    """
    Path in which the database is stored

    Both the index and the data files will be written in this directory.
    """

    @property
    def file_map_file(self) -> Path:
        """
        The file in which the file map is stored

        The file map stores the mapping from file_id
        to file path.

        Returns
        -------
        :
            Path to the file map file
        """
        return self.db_dir / f"filemap{self.backend.ext}"

    @property
    def index_file(self) -> Path:
        """
        The file in which the database's index is stored

        Returns
        -------
        :
            Path to the index file
        """
        return self.db_dir / f"index{self.backend.ext}"

    @property
    def index_file_lock(self) -> filelock.SoftFileLock:
        """Lock for the back-end's index file"""
        return filelock.FileLock(self.index_file_lock_path)

    @property
    def index_file_lock_path(self) -> Path:
        """Path to the lock file for the back-end's index file"""
        return self.index_file.parent / f"{self.index_file.name}.lock"

    @property
    def is_empty(self) -> bool:
        """
        Whether the database is empty or not

        Returns
        -------
        :
            `True` if the database is empty, `False` otherwise
        """
        return not self.index_file.exists()

    def delete(
        self,
        *,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
        progress_results: bool | ProgressLike[Any] | None = None,
        executor: int | concurrent.futures.Executor | None = None,
        progress_parallel_submission: bool | ProgressLike[Any] | None = None,
    ) -> None:
        """
        Delete all data in the database

        Parameters
        ----------
        lock_context_manager
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].

        progress_results
            Progress bar to use to display the results of the deletion's progress.

            If `True`, we simply create a default progress bar.

        executor
            Executor to use for parallel processing.

            If an `int` is supplied, we create an instance of
            [concurrent.futures.ThreadPoolExecutor] with the provided number of workers.

            If not supplied, we do not use parallel processing.

        progress_parallel_submission
            Progress bar to use to display the submission of files to be deleted.

            This only applies when the files are deleted in parallel,
            i.e. `executor` is not `None`.

            If `True`, we simply create a default progress bar.
        """
        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        if isinstance(executor, int):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=executor)

        progress_results_use, progress_parallel_submission_use = (
            figure_out_progress_bars(
                progress_results=progress_results,
                progress_results_default_kwargs=dict(desc="File deletion"),
                executor=executor,
                progress_parallel_submission=progress_parallel_submission,
                progress_parallel_submission_default_kwargs=dict(
                    desc="Submitting files to be deleted"
                ),
            )
        )

        with lock_context_manager:
            apply_op_parallel_progress(
                func_to_call=os.remove,
                iterable_input=self.db_dir.glob(f"*{self.backend.ext}"),
                progress_results=progress_results_use,
                progress_parallel_submission=progress_parallel_submission_use,
                executor=executor,
            )

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
        file_path = self.db_dir / f"{file_id}{self.backend.ext}"

        if file_path.exists():
            raise FileExistsError(file_path)

        return file_path

    def load(  # type: ignore[no-any-unimported] # pix issues
        self,
        selector: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector | None = None,
        *,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
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
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].

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
        if self.is_empty:
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        def idx_obj(inobj: pd.DataFrame) -> pd.DataFrame:
            """
            Do the indexing here

            Return something sensible no matter what the indexer is
            """
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
            index_raw = self.backend.load_index(self.index_file)
            file_map_raw = self.backend.load_file_map(self.file_map_file)
            file_map = file_map_raw.set_index("file_id")["file_path"]

            # Don't need to copy as index_raw is only used internally.
            # The different name is just to help understand the order of operations.
            index = index_raw
            index.index = pd.MultiIndex.from_frame(index_raw)

            index_to_load = idx_obj(index)
            files_to_load: Generator[Path] | tqdm.asyncio.tqdm[Path] = (
                Path(v) for v in file_map[index_to_load["file_id"].unique()]
            )

            if progress:
                try:
                    import tqdm.auto
                except ImportError as exc:
                    raise MissingOptionalDependencyError(  # noqa: TRY003
                        "delete(..., progress=True, ...)", requirement="tqdm"
                    ) from exc

                files_to_load = tqdm.auto.tqdm(files_to_load, desc="Files to load")

            data_l = [self.backend.load_data_file(f) for f in files_to_load]

        loaded = pd.concat(data_l).set_index(index.index.droplevel("file_id").names)

        # Look up only the indexes we want
        # just in case the data we loaded had more than we asked for
        # (because the files aren't saved with exactly the right granularity
        # for the query that has been requested).
        res = idx_obj(loaded)

        if out_columns_type is not None:
            res.columns = res.columns.astype(out_columns_type)

        return res

    def load_metadata(
        self,
        *,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
    ) -> pd.MultiIndex:
        """
        Load the database's metadata

        Parameters
        ----------
        lock_context_manager
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].

        Returns
        -------
        :
            Loaded metadata
        """
        if not self.index_file.exists():
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            db_index = self.backend.load_index(self.index_file)

        res: pd.MultiIndex = pd.MultiIndex.from_frame(db_index).droplevel("file_id")

        return res

    def prepare_for_overwrite(
        self,
        file_ids_db: pd.Series[int],
        data_overwrite: pd.DataFrame,
        file_map: pd.Series[Path],  # type: ignore # pandas confused about ability to support Path
    ) -> tuple[pd.DataFrame, pd.Series[Path]]:  # type: ignore # pandas confused about ability to support Path
        """
        Prepare to overwrite data that is already in the database.

        The data that is being overwritten will also be cleared from the database,
        so `data_overwrite` can be written after calling this method without issue.

        Unless you really know what you're doing,
        you probably shouldn't be using this directly.

        Parameters
        ----------
        file_ids_db
            File IDs that are already in the database

        data_overwrite
            Data that will overwrite data that is already in the database.

        file_map
            Existing map from file IDs to files.

        Returns
        -------
        index_out  :
            Updated index to use.

        file_map_out :
            Updated file map to use.
        """
        overwrite_loc = multi_index_match(file_ids_db.index, data_overwrite.index)  # type: ignore # pandas confused about index type
        file_ids_remove = set(file_ids_db[overwrite_loc])
        file_ids_keep = set(file_ids_db[~overwrite_loc])
        file_ids_overlap = file_ids_remove.intersection(file_ids_keep)

        file_map_out = file_map.copy()

        if not file_ids_overlap:
            # Nice and simple, just remove the old files that we're going to overwrite
            index_out = file_ids_db[~overwrite_loc].reset_index()
            for rfid in file_ids_remove:
                os.remove(file_map_out.pop(rfid))

        else:
            # More complicated: for some files,
            # some of the data needs to be removed
            # while other parts need to be kept.
            # Hence we have to re-write that data into files
            # that are separate from the data we want to keep
            # before we can continue.
            file_ids_out = file_ids_db.copy()
            for overlap_fid in file_ids_overlap:
                overlap_file = file_map_out.pop(overlap_fid)

                overlap_file_data_raw = self.backend.load_data_file(overlap_file)
                overlap_file_data = overlap_file_data_raw.set_index(
                    file_ids_db.index.names
                )

                data_not_being_overwritten = overlap_file_data.loc[
                    ~multi_index_match(overlap_file_data.index, data_overwrite.index)  # type: ignore # pandas confused about index type
                ]

                # Ensure we use a file ID we haven't already used
                data_not_being_overwritten_file_id = (
                    max(file_map_out.index.max(), file_map.index.max()) + 1
                )
                data_not_being_overwritten_file_path = self.get_new_data_file_path(
                    file_id=data_not_being_overwritten_file_id
                )

                # Re-write the data we want to keep
                self.backend.save_data(
                    data_not_being_overwritten, data_not_being_overwritten_file_path
                )

                # Update the file map (already popped the old file above)
                file_map_out[data_not_being_overwritten_file_id] = (
                    data_not_being_overwritten_file_path
                )

                # Update the file ids of the data we're keeping
                file_ids_out.loc[
                    multi_index_match(
                        file_ids_out.index,  # type: ignore # pandas confused about index type
                        data_not_being_overwritten.index,  # type: ignore # pandas confused about index type
                    )
                ] = data_not_being_overwritten_file_id

                # Remove the rows that still refer to the data we're dropping
                file_ids_out = file_ids_out.loc[file_ids_out != overlap_fid]

                # Remove the file that contained the data
                os.remove(overlap_file)

            index_out = file_ids_out.reset_index()

        return index_out, file_map_out

    def save(
        self,
        data: pd.DataFrame,
        *,
        allow_overwrite: bool = False,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
    ) -> None:
        """
        Save data into the database

        This saves all of the data in a single file.
        If you want to put the data into different files,
        group `data` before calling [save][(c)].

        Parameters
        ----------
        data
            Data to add to the database

        allow_overwrite
            Should overwrites of data that is already in the database be allowed?

        lock_context_manager
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].
        """
        # TODO: add check that data has no duplicates in its index
        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            if self.is_empty:
                file_id = 0
                data_file = self.get_new_data_file_path(file_id=file_id)
                index_db = data.index.to_frame(index=False)
                index_db["file_id"] = file_id
                file_map_db = pd.Series({file_id: data_file}, name="file_path")
                file_map_db.index.name = "file_id"

            else:
                index_db = self.backend.load_index(self.index_file)
                file_map_db = self.backend.load_file_map(self.file_map_file).set_index(
                    "file_id"
                )["file_path"]
                metadata_db = pd.MultiIndex.from_frame(index_db).droplevel("file_id")

                file_id = index_db["file_id"].astype(int).max() + 1
                data_file = self.get_new_data_file_path(file_id=file_id)

                file_map_db[file_id] = data_file
                data_index = data.index.to_frame(index=False)
                data_index["file_id"] = file_id

                data_to_write_already_in_db = multi_index_lookup(data, metadata_db)
                if data_to_write_already_in_db.empty:
                    # No clashes, so we can simply concatenate
                    index_db = pd.concat([index_db, data_index])

                else:
                    if not allow_overwrite:
                        raise AlreadyInDBError(
                            already_in_db=data_to_write_already_in_db
                        )

                    index_db_keep, file_map_db = self.prepare_for_overwrite(
                        file_ids_db=index_db.set_index(metadata_db.names)["file_id"],
                        data_overwrite=data_to_write_already_in_db,
                        file_map=file_map_db,
                    )

                    index_db = pd.concat([index_db_keep, data_index])

            self.backend.save_database(
                index=index_db,
                index_file=self.index_file,
                file_map=file_map_db,
                file_map_file=self.file_map_file,
                data=data,
                data_file=data_file,
            )


__all__ = [
    "AlreadyInDBError",
    "CSVBackend",
    "EmptyDBError",
    "FeatherBackend",
    "OpenSCMDB",
    "OpenSCMDBBackend",
    "netCDFBackend",
]
