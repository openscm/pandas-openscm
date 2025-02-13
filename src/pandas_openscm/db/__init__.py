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
import numpy as np
import pandas as pd
from attrs import define

from pandas_openscm.db.csv import CSVBackend
from pandas_openscm.db.feather import FeatherBackend
from pandas_openscm.db.netcdf import netCDFBackend
from pandas_openscm.indexing import mi_loc, multi_index_match
from pandas_openscm.parallelisation import (
    ProgressLike,
    apply_op_parallel_progress,
    figure_out_progress_bars,
)

if TYPE_CHECKING:
    import pandas_indexing as pix  # type: ignore # see https://github.com/coroa/pandas-indexing/pull/63


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


@define
class ReWriteAction:
    """Description of a re-write action"""

    from_file: Path
    """File from which to load the data"""

    to_file: Path
    """File in which to write the re-written data"""

    locator: pd.MultiIndex
    """Locator which specifies which data to re-write"""


@define
class MovePlan:
    """Plan for how to move data to make way for an overwrite"""

    moved_index: pd.DataFrame
    """The index once all the data has been moved"""

    moved_file_map: pd.Series[Path]  # type: ignore # pandas confused about ability to support Path
    """The file map once all the data has been moved"""

    rewrite_actions: tuple[ReWriteAction, ...] | None
    """The re-write actions which need to be performed"""

    delete_paths: tuple[Path, ...] | None
    """Paths which can be deleted (after the data has been moved)"""


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
        progress_results: bool | ProgressLike | None = None,
        executor: int | concurrent.futures.Executor | None = None,
        progress_parallel_submission: bool | ProgressLike | None = None,
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

            If you are interested in parallel processing,
            the docs in [parallelisation][(p).] might be worth reading first.

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
            # Threading by default as deletion is I/O bound
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=executor)

        progress_results_use, progress_parallel_submission_use = (
            figure_out_progress_bars(
                progress_results=progress_results,
                progress_results_default_kwargs=dict(desc="File deletion"),
                executor=executor,
                progress_parallel_submission=progress_parallel_submission,
                progress_parallel_submission_default_kwargs=dict(
                    desc="Submitting files to the parallel executor"
                ),
            )
        )

        iterable_input: Generator[Path] | list[Path] = self.db_dir.glob(
            f"*{self.backend.ext}"
        )
        if progress_results_use is not None:
            # Wrap in list to force the length to be available to any progress bar
            iterable_input = list(iterable_input)

        with lock_context_manager:
            apply_op_parallel_progress(
                func_to_call=os.remove,
                iterable_input=iterable_input,
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

    def load(  # type: ignore[no-any-unimported] # type ignore b/c of pix issues # noqa: PLR0913
        self,
        selector: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector | None = None,
        *,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
        out_columns_type: type | None = None,
        progress_results: bool | ProgressLike | None = None,
        executor: int | concurrent.futures.Executor | None = None,
        progress_parallel_submission: bool | ProgressLike | None = None,
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

        out_columns_type
            Type to set the output columns to.

            If not supplied, we don't set the output columns' type.

        progress_results
            Progress bar to use to display the results of the deletion's progress.

            If `True`, we simply create a default progress bar.

        executor
            Executor to use for parallel processing.

            If you are interested in parallel processing,
            the docs in [parallelisation][(p).] might be worth reading first.

            If an `int` is supplied, we create an instance of
            [concurrent.futures.ProcessPoolExecutor]
            with the provided number of workers.
            A process pool seems to be the sensible default from our experimentation,
            but it is not a universally better choice.
            If you need something else because of how your database is set up,
            simply pass an executor
            rather than using the shortcut of passing an integer.

            If not supplied, we do not use parallel processing.

        progress_parallel_submission
            Progress bar to use to display the submission of files to be deleted.

            This only applies when the files are deleted in parallel,
            i.e. `executor` is not `None`.

            If `True`, we simply create a default progress bar.

        Returns
        -------
        :
            Loaded data
        """
        if self.is_empty:
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        if isinstance(executor, int):
            # Process pool by default as basic tests suggest
            # that reading is CPU-bound.
            # See the docs for nuance though.
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=executor)

        progress_results_use, progress_parallel_submission_use = (
            figure_out_progress_bars(
                progress_results=progress_results,
                progress_results_default_kwargs=dict(desc="Files to load"),
                executor=executor,
                progress_parallel_submission=progress_parallel_submission,
                progress_parallel_submission_default_kwargs=dict(
                    desc="Submitting files to the parallel executor"
                ),
            )
        )

        with lock_context_manager:
            index_raw = self.backend.load_index(self.index_file)
            file_map_raw = self.backend.load_file_map(self.file_map_file)
            file_map = file_map_raw.set_index("file_id")["file_path"]

            # Don't need to copy as index_raw is only used internally.
            # The different name is just to help understand the order of operations.
            index = index_raw
            index.index = pd.MultiIndex.from_frame(index_raw)

            index_to_load = index
            if selector is not None:
                index_to_load = mi_loc(index_to_load, selector)

            files_to_load: Generator[Path] | list[Path] = (
                Path(v) for v in file_map[index_to_load["file_id"].unique()]
            )
            if progress_results_use is not None:
                # Wrap in list to force the length to be available to any progress bar
                files_to_load = list(files_to_load)

            data_l = apply_op_parallel_progress(
                func_to_call=self.backend.load_data_file,
                iterable_input=files_to_load,
                progress_results=progress_results_use,
                progress_parallel_submission=progress_parallel_submission_use,
                executor=executor,
            )

        loaded = pd.concat(data_l).set_index(index.index.droplevel("file_id").names)

        # Look up only the indexes we want
        # just in case the data we loaded had more than we asked for
        # (because the files aren't saved with exactly the right granularity
        # for the query that has been requested).
        res = loaded
        if selector is not None:
            res = mi_loc(res, selector)

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

    def make_move_plan(
        self,
        index_start: pd.DataFrame,
        file_map_start: pd.Series[Path],  # type: ignore # pandas confused about ability to support Path
        data_to_write: pd.DataFrame,
    ) -> MovePlan:
        """
        Make a plan for moving data around to make room for new data

        Parameters
        ----------
        index_start
            The starting index

        file_map_start
            The starting file map

        data_to_write
            Data that is going to be written in the database

        Returns
        -------
        :
            Plan for moving data to make room for the new data
        """
        index_start_mi = pd.MultiIndex.from_frame(index_start)
        in_data_to_write = pd.Series(
            multi_index_match(index_start_mi, data_to_write.index),
            index=index_start_mi,
        )

        grouper = in_data_to_write.groupby("file_id")
        no_overwrite = ~grouper.apply(np.any)
        if no_overwrite.all():
            # Can push this all into the layer above
            # new_file_id = index_start["file_id"].max() + 1
            #
            # file_map_out = file_map_start.copy()
            # file_map_out.loc[new_file_id] = self.get_new_data_file_path(new_file_id)
            #
            # index_data_to_write = data_to_write.index.to_frame(index=False)
            # index_data_to_write["file_id"] = new_file_id
            #
            # moved_index = pd.concat([index_start, index_data_to_write])

            return MovePlan(
                moved_index=index_start,
                moved_file_map=file_map_start,
                rewrite_actions=None,
                delete_paths=None,
            )

        full_overwrite = grouper.apply(np.all)
        partial_overwrite = ~(full_overwrite | no_overwrite)
        if not partial_overwrite.any():
            # Only keep the bits of the index which won't be overwritten
            delete_file_ids = full_overwrite.index[full_overwrite]
            delete_paths = file_map_start.loc[delete_file_ids]
            moved_index = index_start[~index_start["file_id"].isin(delete_file_ids)]
            file_map_out = file_map_start.loc[moved_index["file_id"]]

            return MovePlan(
                moved_index=moved_index,
                moved_file_map=file_map_out,
                rewrite_actions=None,
                delete_paths=tuple(delete_paths),
            )

        # # TODO: move this into the layer above,
        # # where the re-write will actually happen
        # if warn_on_partial_overwrite:
        #     msg = (
        #         "Overwriting the data will require re-writing. "
        #         "This may be slow. "
        #         "If that is an issue, the only way to solve it is to "
        #         "update your workflow to ensure that you are not overwriting data "
        #         "or are only overwriting entire files."
        #     )
        #     warnings.warn(msg)

        to_rewrite = partial_overwrite & ~in_data_to_write

        full_overwrite_file_ids = full_overwrite.index[full_overwrite]
        file_ids_to_delete = np.union1d(
            full_overwrite_file_ids,
            partial_overwrite.index[partial_overwrite],
        )
        delete_paths = file_map_start.loc[file_ids_to_delete]

        index_moved = in_data_to_write[
            in_data_to_write
            & ~in_data_to_write.index.get_level_values("file_id").isin(
                full_overwrite_file_ids
            )
        ].index.to_frame(index=False)

        file_id_map = {}
        file_map_out = file_map_start[no_overwrite].copy()
        max_file_id_start = file_map_start.index.max()
        rewrite_actions_l = []
        for increment, (file_id_old, fiddf) in enumerate(
            to_rewrite.loc[to_rewrite].groupby("file_id")
        ):
            new_file_id = max_file_id_start + 1 + increment

            file_map_out.loc[new_file_id] = self.get_new_data_file_path(new_file_id)

            rewrite_actions_l.append(
                ReWriteAction(
                    from_file=file_map_start.loc[file_id_old],
                    to_file=file_map_out.loc[new_file_id],
                    locator=fiddf.index.droplevel("file_id"),
                )
            )
            file_id_map[file_id_old] = new_file_id

        index_moved["file_id"] = index_moved["file_id"].map(file_id_map)
        if index_moved["file_id"].isnull().any():
            # Something has gone wrong, everything should be remapped somewhere
            raise AssertionError

        moved_index = pd.concat(
            [
                # Bits of the index which won't be overwritten
                index_start[~index_start["file_id"].isin(file_ids_to_delete)],
                # Bits of the index which are being re-written
                index_moved,
            ]
        )
        res = MovePlan(
            moved_index=moved_index,
            moved_file_map=file_map_out,
            rewrite_actions=tuple(rewrite_actions_l),
            delete_paths=tuple(delete_paths),
        )

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

                data_to_write_already_in_db = mi_loc(data, metadata_db)
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
