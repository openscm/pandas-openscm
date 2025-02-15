"""
Database definition
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import warnings
from collections.abc import Iterable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import filelock
import numpy as np
import pandas as pd
from attrs import define

from pandas_openscm.db.csv import CSVBackend
from pandas_openscm.db.feather import FeatherBackend
from pandas_openscm.db.netcdf import netCDFBackend
from pandas_openscm.index_manipulation import update_index_from_candidates
from pandas_openscm.indexing import mi_loc, multi_index_match
from pandas_openscm.parallelisation import (
    ParallelOpConfig,
    ProgressLike,
    apply_op_parallel_progress,
    get_tqdm_auto,
)

if TYPE_CHECKING:
    import pandas.core.groupby.generic
    import pandas.core.indexes.frozen
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

    preserves_index: bool
    """
    Whether this backend preserves the index
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

    def save_file_map(
        self,
        file_map: pd.Series[Path],  # type: ignore # pandas confused about what it supports
        file_map_file: Path,
    ) -> None:
        """
        Save the file map

        This is a low-level method
        that just handles the specifics of serialising the file map to disk.
        Working out what data to save and in what path
        should happen in higher-level functions.

        Parameters
        ----------
        file_map
            File map to save

        file_map_file
            File in which to save the file map
        """

    def save_index(
        self,
        index: pd.DataFrame,
        index_file: Path,
    ) -> None:
        """
        Save the index

        This is a low-level method
        that just handles the specifics of serialising the index to disk.
        Working out what data to save and in what path
        should happen in higher-level functions.

        Parameters
        ----------
        index
            Index to save

        index_file
            File in which to save the index
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
        parallel_op_config: ParallelOpConfig | None = None,
        progress: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """
        Delete all data in the database

        Parameters
        ----------
        lock_context_manager
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].

        parallel_op_config
            Configuration for executing the operation in parallel with progress bars

            If not supplied, we use the values of `progress` and `max_workers`.

        progress
            Should progress bar(s) be used to display the progress of the deletion?

            Only used if `parallel_op_config` is `None`.

        max_workers
            Maximum number of workers to use for parallel processing.

            If supplied, we create an instance of
            [concurrent.futures.ThreadPoolExecutor][]
            with the provided number of workers
            (a thread pool makes sense as deletion is I/O-bound).

            If not supplied, the deletions are executed serially.

            Only used if `parallel_op_config` is `None`.
        """
        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            files_to_delete = self.db_dir.glob(f"*{self.backend.ext}")
            delete_files(
                files_to_delete=files_to_delete,
                parallel_op_config=parallel_op_config,
                progress=progress,
                max_workers=max_workers,
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
        parallel_op_config: ParallelOpConfig | None = None,
        progress: bool = False,
        max_workers: int | None = None,
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

        parallel_op_config
            Configuration for executing the operation in parallel with progress bars

            If not supplied, we use the values of `progress` and `max_workers`.

        progress
            Should progress bar(s) be used to display the progress of the deletion?

            Only used if `parallel_op_config` is `None`.

        max_workers
            Maximum number of workers to use for parallel processing.

            If supplied, we create an instance of
            [concurrent.futures.ProcessPoolExecutor][]
            with the provided number of workers.
            A process pool seems to be the sensible default from our experimentation,
            but it is not a universally better choice.
            If you need something else because of how your database is set up,
            simply pass `parallel_op_config`
            rather than using the shortcut of passing `max_workers`.

            If not supplied, the loading is executed serially.

            Only used if `parallel_op_config` is `None`.

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

        with lock_context_manager:
            # index_raw = self.backend.load_index(self.index_file)
            #
            # # Don't need to copy as index_raw is only used internally.
            # # The different name is just to help understand the order of operations.
            # index = index_raw
            # index.index = pd.MultiIndex.from_frame(index_raw)
            #
            # index_to_load = index
            file_map = self.load_file_map(
                # Already have the lock
                lock_context_manager=contextlib.nullcontext(None)
            )
            index = self.load_index(
                # Already have the lock
                lock_context_manager=contextlib.nullcontext(None)
            )
            if selector is None:
                index_to_load = index
            else:
                index_to_load = mi_loc(index, selector)

            files_to_load = (
                Path(v) for v in file_map[index_to_load["file_id"].unique()]
            )
            loaded_l = load_files(
                files_to_load=files_to_load,
                backend=self.backend,
                parallel_op_config=parallel_op_config,
                progress=progress,
                max_workers=max_workers,
            )

        res = pd.concat(loaded_l)
        if not self.backend.preserves_index:
            index_names: pandas.core.indexes.frozen.FrozenList = index.index.names  # type: ignore # pandas type hints wrong
            res = update_index_from_candidates(res, index_names.difference({"file_id"}))

        # Look up only the indexes we want
        # just in case the data we loaded had more than we asked for
        # (because the files aren't saved with exactly the right granularity
        # for the query that has been requested).
        if selector is not None:
            res = mi_loc(res, selector)

        if out_columns_type is not None:
            res.columns = res.columns.astype(out_columns_type)

        return res

    def load_file_map(
        self,
        *,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
    ) -> pd.Series[Path]:  # type: ignore # pandas type hints confused about what they support
        """
        Load the file map

        Parameters
        ----------
        lock_context_manager
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].

        Returns
        -------
        :
            Map from file ID to file path

        Raises
        ------
        EmptyDBError
            The database is empty
        """
        if self.is_empty:
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            file_map_raw = self.backend.load_file_map(self.file_map_file)
            if not self.backend.preserves_index:
                file_map_indexed = file_map_raw.set_index("file_id")
            else:
                file_map_indexed = file_map_raw

            file_map = file_map_indexed["file_path"]

        return file_map

    def load_index(
        self,
        *,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
    ) -> pd.DataFrame:
        """
        Load the index

        Parameters
        ----------
        lock_context_manager
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].

        Returns
        -------
        :
            Database index

        Raises
        ------
        EmptyDBError
            The database is empty
        """
        if self.is_empty:
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            index = self.backend.load_index(self.index_file)

        if not self.backend.preserves_index:
            index = index.set_index(index.columns.difference(["file_id"]).to_list())

        return index

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
            db_index = self.load_index(
                # Already have the lock
                lock_context_manager=contextlib.nullcontext(None)
            )

        if not isinstance(db_index.index, pd.MultiIndex):  # pragma: no cover
            # Should be impossible to get here
            raise TypeError

        res: pd.MultiIndex = db_index.index

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
        if not isinstance(data_to_write.index, pd.MultiIndex):
            raise TypeError

        in_data_to_write = pd.Series(
            multi_index_match(index_start.index, data_to_write.index),  # type: ignore # pandas type hints confused
            index=index_start.set_index("file_id", append=True).index,
        )

        grouper = in_data_to_write.groupby("file_id")
        no_overwrite = ~grouper.apply(np.any)
        if no_overwrite.all():
            return MovePlan(
                moved_index=index_start,
                moved_file_map=file_map_start,
                rewrite_actions=None,
                delete_paths=None,
            )

        full_overwrite: pd.Series[bool] = grouper.apply(np.all)  # type: ignore # pandas type hints confused
        partial_overwrite = ~(full_overwrite | no_overwrite)
        if not partial_overwrite.any():
            # Only keep the bits of the index which won't be overwritten
            delete_file_ids = full_overwrite.index[full_overwrite]
            delete_paths = file_map_start.loc[delete_file_ids]
            moved_index = index_start[~index_start["file_id"].isin(delete_file_ids)]
            file_map_out = file_map_start.loc[moved_index["file_id"].unique()]

            return MovePlan(
                moved_index=moved_index,
                moved_file_map=file_map_out,
                rewrite_actions=None,
                delete_paths=tuple(delete_paths),
            )

        to_rewrite = partial_overwrite & ~in_data_to_write

        full_overwrite_file_ids = full_overwrite.index[full_overwrite]
        partial_overwrite_file_ids = partial_overwrite.index[partial_overwrite]
        file_ids_to_delete = np.union1d(
            full_overwrite_file_ids,
            partial_overwrite_file_ids,
        )
        delete_paths = file_map_start.loc[file_ids_to_delete]

        index_moved = in_data_to_write[
            ~in_data_to_write
            & in_data_to_write.index.get_level_values("file_id").isin(
                partial_overwrite_file_ids
            )
        ].reset_index("file_id")[["file_id"]]

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

    def save(  # noqa: PLR0913
        self,
        data: pd.DataFrame,
        *,
        lock_context_manager: contextlib.AbstractContextManager[Any] | None = None,
        groupby: list[str] | None = None,
        allow_overwrite: bool = False,
        warn_on_partial_overwrite: bool = True,
        progress_grouping: ProgressLike | None = None,
        parallel_op_config_save: ParallelOpConfig | None = None,
        parallel_op_config_delete: ParallelOpConfig | None = None,
        parallel_op_config_rewrite: ParallelOpConfig | None = None,
        progress: bool = False,
        max_workers: int | None = None,
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

        lock_context_manager
            Context manager to use to acquire the lock file.

            If not supplied, we use
            [`self.index_file_lock.acquire`][(c)].

        groupby
            Metadata columns to use to group the data.

            If not supplied, we save all the data in a single file.

        allow_overwrite
            Should overwrites of data that is already in the database be allowed?

            If this is `True`, there is a risk that, if interrupted halfway through,
            you can end up with duplicate data in your database
            or some other odd broken state.

        warn_on_partial_overwrite
            Should a warning be raised if a partial overwrite will occur?

            This is on by default so that users
            are warned about the slow operation of re-writing.

        progress_grouping
            Progress bar to use when grouping the data

            If not supplied, we use the values of `progress` and `max_workers`.

        parallel_op_config_save
            Parallel op configuration for executing save operations

            If not supplied, we use the values of `progress` and `max_workers`.

        parallel_op_config_delete
            Parallel op configuration for executing any needed delete operations

            If not supplied, we use the values of `progress` and `max_workers`.

        parallel_op_config_rewrite
            Parallel op configuration for executing any needed re-write operations

            If not supplied, we use the values of `progress` and `max_workers`.

        progress
            Should progress bar(s) be used to display the progress of the various steps?

            Only used if the corresponding `parallel_op_config_*` variable
            for the operation is `None`.

        max_workers
            Maximum number of workers to use for parallel processing.

            If supplied, we create instances of
            [concurrent.futures.Executor][]
            with the provided number of workers
            (the exact kind of executor depends on the operation).

            If not supplied, the operations are executed serially.

            Only used if the corresponding `parallel_op_config_*` variable
            for the operation is `None`.
        """
        if not isinstance(data.index, pd.MultiIndex):
            msg = f"data must have a MultiIndex, received {type(data.index)=}"
            raise TypeError(msg)

        if data.index.duplicated().any():
            duplicate_rows = data.index.duplicated(keep=False)
            duplicates = data.loc[duplicate_rows, :]
            # TODO: more helpful error message
            raise ValueError(duplicates)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            if self.is_empty:
                save_data(
                    data,
                    db=self,
                    min_file_id=0,
                    groupby=groupby,
                    progress_grouping=progress_grouping,
                    parallel_op_config=parallel_op_config_save,
                    progress=progress,
                    max_workers=max_workers,
                )

                return

            file_map_db = self.load_file_map(
                # Already have the lock
                lock_context_manager=contextlib.nullcontext(None)
            )
            index_db = self.load_index(
                # Already have the lock
                lock_context_manager=contextlib.nullcontext(None)
            )
            if not allow_overwrite:
                if not isinstance(index_db.index, pd.MultiIndex):  # pragma: no cover
                    # Should be impossible to get here
                    raise TypeError

                overwrite_required = multi_index_match(data.index, index_db.index)

                if overwrite_required.any():
                    data_to_write_already_in_db = data.loc[overwrite_required, :]
                    raise AlreadyInDBError(already_in_db=data_to_write_already_in_db)

            move_plan = self.make_move_plan(
                index_start=index_db, file_map_start=file_map_db, data_to_write=data
            )

            # As needed, re-write files without deleting the old files
            if move_plan.rewrite_actions is not None:
                if warn_on_partial_overwrite:
                    msg = (
                        "Overwriting the data will require re-writing. "
                        "This may be slow. "
                        "If that is an issue, the way to solve it "
                        "is to update your workflow to ensure "
                        "that you are not overwriting data "
                        "or are only overwriting entire files."
                    )
                    warnings.warn(msg)

                rewrite_files(
                    move_plan.rewrite_actions,
                    backend=self.backend,
                    parallel_op_config=parallel_op_config_rewrite,
                    progress=progress,
                    max_workers=max_workers,
                )

            # Write the new data
            current_largest_file_id = file_map_db.index.max()
            if not move_plan.moved_file_map.empty:
                current_largest_file_id = max(
                    move_plan.moved_file_map.index.max(), current_largest_file_id
                )

            min_file_id = current_largest_file_id + 1

            save_data(
                data,
                db=self,
                index_non_data=move_plan.moved_index,
                file_map_non_data=move_plan.moved_file_map,
                min_file_id=min_file_id,
                groupby=groupby,
                progress_grouping=progress_grouping,
                parallel_op_config=parallel_op_config_save,
                progress=progress,
                max_workers=max_workers,
            )

            # As needed, delete files.
            # We delete files last to minimise the risk of losing data
            # (might end up with double if we get interrupted here,
            # but that is better than zero).
            if move_plan.delete_paths is not None:
                delete_files(
                    files_to_delete=move_plan.delete_paths,
                    parallel_op_config=parallel_op_config_delete,
                    progress=progress,
                    max_workers=max_workers,
                )


def delete_files(
    files_to_delete: Iterable[Path],
    parallel_op_config: ParallelOpConfig | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Delete a number of files

    Parameters
    ----------
    files_to_delete
        Files to delete

    parallel_op_config
        Configuration for executing the operation in parallel with progress bars

        If not supplied, we use the values of `progress` and `max_workers`.

    progress
        Should progress bar(s) be used to display the progress of the deletion?

        Only used if `parallel_op_config` is `None`.

    max_workers
        Maximum number of workers to use for parallel processing.

        If supplied, we create an instance of
        [concurrent.futures.ThreadPoolExecutor] with the provided number of workers
        (a thread pool makes sense as deletion is I/O-bound).

        If not supplied, the deletions are executed serially.

        Only used if `parallel_op_config` is `None`.
    """
    iterable_input: Iterable[Path] | list[Path] = files_to_delete

    # Stick the whole thing in a try finally block so we shutdown
    # the parallel pool, even if interrupted, if we created it.
    try:
        if parallel_op_config is None:
            parallel_op_config_use = ParallelOpConfig.from_user_facing(
                progress=progress,
                progress_results_kwargs=dict(desc="File deletion"),
                progress_parallel_submission_kwargs=dict(
                    desc="Submitting files to the parallel executor"
                ),
                max_workers=max_workers,
                parallel_pool_cls=concurrent.futures.ThreadPoolExecutor,
            )
        else:
            parallel_op_config_use = parallel_op_config

        if parallel_op_config_use.progress_results is not None:
            # Wrap in list to force the length to be available to any progress bar.
            # This might be the wrong decision in a weird edge case,
            # but it's convenient enough that I'm willing to take that risk
            iterable_input = list(iterable_input)

        apply_op_parallel_progress(
            func_to_call=os.remove,
            iterable_input=iterable_input,
            parallel_op_config=parallel_op_config_use,
        )

    finally:
        if parallel_op_config_use.executor_created_in_class_method:
            if parallel_op_config_use.executor is None:  # pragma: no cover
                # Should be impossible to get here
                raise AssertionError

            parallel_op_config_use.executor.shutdown()


def load_files(
    files_to_load: Iterable[Path],
    backend: OpenSCMDBBackend,
    parallel_op_config: ParallelOpConfig | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, ...]:
    """
    Load a number of files

    Parameters
    ----------
    files_to_load
        Files to load

    backend
        Backend to use to load the files

    parallel_op_config
        Configuration for executing the operation in parallel with progress bars

        If not supplied, we use the values of `progress` and `max_workers`.

    progress
        Should progress bar(s) be used to display the progress of the deletion?

        Only used if `parallel_op_config` is `None`.

    max_workers
        Maximum number of workers to use for parallel processing.

        If supplied, we create an instance of
        [concurrent.futures.ProcessPoolExecutor][]
        with the provided number of workers.
        A process pool seems to be the sensible default from our experimentation,
        but it is not a universally better choice.
        If you need something else because of how your database is set up,
        simply pass `parallel_op_config`
        rather than using the shortcut of passing `max_workers`.

        If not supplied, the loading is executed serially.

        Only used if `parallel_op_config` is `None`.
    """
    iterable_input: Iterable[Path] | list[Path] = files_to_load

    # Stick the whole thing in a try finally block so we shutdown
    # the parallel pool, even if interrupted, if we created it.
    try:
        if parallel_op_config is None:
            parallel_op_config_use = ParallelOpConfig.from_user_facing(
                progress=progress,
                progress_results_kwargs=dict(desc="File loading"),
                progress_parallel_submission_kwargs=dict(
                    desc="Submitting files to the parallel executor"
                ),
                max_workers=max_workers,
                # Process pool by default as basic tests suggest
                # that reading is CPU-bound.
                # See the docs for nuance though.
                parallel_pool_cls=concurrent.futures.ProcessPoolExecutor,
            )
        else:
            parallel_op_config_use = parallel_op_config

        if parallel_op_config_use.progress_results is not None:
            # Wrap in list to force the length to be available to any progress bar.
            # This might be the wrong decision in a weird edge case,
            # but it's convenient enough that I'm willing to take that risk
            iterable_input = list(iterable_input)

        res = apply_op_parallel_progress(
            func_to_call=backend.load_data_file,
            iterable_input=iterable_input,
            parallel_op_config=parallel_op_config_use,
        )

    finally:
        if parallel_op_config_use.executor_created_in_class_method:
            if parallel_op_config_use.executor is None:  # pragma: no cover
                # Should be impossible to get here
                raise AssertionError

            parallel_op_config_use.executor.shutdown()

    return res


def rewrite_file(
    rewrite_action: ReWriteAction,
    backend: OpenSCMDBBackend,
) -> None:
    """
    Re-write a file

    Parameters
    ----------
    rewrite_action
        Re-write action to perform

    backend
        Back-end to use for reading and writing data
    """
    data_all = backend.load_data_file(rewrite_action.from_file)
    if not backend.preserves_index:
        rewrite_action_names: pandas.core.indexes.frozen.FrozenList = (
            rewrite_action.locator.names  # type: ignore # pandas type hints wrong
        )
        data_all = update_index_from_candidates(
            data_all,
            rewrite_action_names,
        )

    data_rewrite = mi_loc(data_all, rewrite_action.locator)
    backend.save_data(data_rewrite, rewrite_action.to_file)


def rewrite_files(
    rewrite_actions: Iterable[ReWriteAction],
    backend: OpenSCMDBBackend,
    parallel_op_config: ParallelOpConfig | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Re-write a number of files

    Parameters
    ----------
    rewrite_actions
        Re-write actions to perform

    backend
        Backend to use to load and write the files

    parallel_op_config
        Configuration for executing the operation in parallel with progress bars

        If not supplied, we use the values of `progress` and `max_workers`.

    progress
        Should progress bar(s) be used to display the progress of the deletion?

        Only used if `parallel_op_config` is `None`.

    max_workers
        Maximum number of workers to use for parallel processing.

        If supplied, we create an instance of
        [concurrent.futures.ProcessPoolExecutor][]
        with the provided number of workers.
        A process pool seems to be the sensible default from our experimentation,
        but it is not a universally better choice.
        If you need something else because of how your database is set up,
        simply pass `parallel_op_config`
        rather than using the shortcut of passing `max_workers`.

        If not supplied, the loading is executed serially.

        Only used if `parallel_op_config` is `None`.
    """
    iterable_input: Iterable[ReWriteAction] | list[ReWriteAction] = rewrite_actions

    # Stick the whole thing in a try finally block so we shutdown
    # the parallel pool, even if interrupted, if we created it.
    try:
        if parallel_op_config is None:
            parallel_op_config_use = ParallelOpConfig.from_user_facing(
                progress=progress,
                progress_results_kwargs=dict(desc="File re-writing"),
                progress_parallel_submission_kwargs=dict(
                    desc="Submitting files to the parallel executor"
                ),
                max_workers=max_workers,
                # Process pool by default as basic tests suggest
                # that reading, therefore re-writing, is CPU-bound.
                # See the docs for nuance though.
                parallel_pool_cls=concurrent.futures.ProcessPoolExecutor,
            )
        else:
            parallel_op_config_use = parallel_op_config

        if parallel_op_config_use.progress_results is not None:
            # Wrap in list to force the length to be available to any progress bar.
            # This might be the wrong decision in a weird edge case,
            # but it's convenient enough that I'm willing to take that risk
            iterable_input = list(iterable_input)

        apply_op_parallel_progress(
            func_to_call=rewrite_file,
            iterable_input=iterable_input,
            parallel_op_config=parallel_op_config_use,
            backend=backend,
        )

    finally:
        if parallel_op_config_use.executor_created_in_class_method:
            if parallel_op_config_use.executor is None:  # pragma: no cover
                # Should be impossible to get here
                raise AssertionError

            parallel_op_config_use.executor.shutdown()


class DBFileType(Enum):
    """
    Type of a database file

    Really just a helper for [save_data][(m).]
    """

    DATA = auto()
    INDEX = auto()
    FILE_MAP = auto()


@define
class SaveAction:
    """A database save action"""

    info: pd.DataFrame | pd.Series[Any]
    """Information to save"""

    info_kind: DBFileType
    """The kind of information that this is"""

    save_path: Path
    """Path in which to save the information"""


def save_data(  # noqa: PLR0913
    data: pd.DataFrame,
    db: OpenSCMDB,
    index_non_data: pd.DataFrame | None = None,
    file_map_non_data: pd.Series[Path] | None = None,  # type: ignore # pandas type hints doesn't know what it supports
    min_file_id: int = 0,
    groupby: list[str] | None = None,
    progress_grouping: ProgressLike | None = None,
    parallel_op_config: ParallelOpConfig | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Save data

    Parameters
    ----------
    data
        Data to save

    db
        Database in which to save the data

    index_non_data
        Index that is already in the database but isn't related to data.

        If supplied, this is combined with the index generated for `data`
        before we write the database's index.

    file_map_non_data
        File map that is already in the database but isn't related to data.

        If supplied, this is combined with the file map generated for `data`
        before we write the database's file map.

    min_file_id
        Minimum file ID to assign to save data chunks

    groupby
        Metadata columns to use to group the data.

        If not supplied, we save all the data in a single file.

    progress_grouping
        Progress bar to use when grouping the data

    parallel_op_config
        Configuration for executing the operation in parallel with progress bars

        If not supplied, we use the values of `progress` and `max_workers`.

    progress
        Should progress bar(s) be used to display the progress of the saving?

        Only used if `progress_grouping` is `None` or `parallel_op_config` is `None`.

    max_workers
        Maximum number of workers to use for parallel processing.

        If supplied, we create an instance of
        [concurrent.futures.ProcessPoolExecutor][]
        with the provided number of workers.
        A process pool seems to be the sensible default from our experimentation,
        but it is not a universally better choice.
        If you need something else because of how your database is set up,
        simply pass `parallel_op_config`
        rather than using the shortcut of passing `max_workers`.

        If not supplied, the saving is executed serially.

        Only used if `parallel_op_config` is `None`.
    """
    if groupby is None:
        # Write as a single file
        grouper: (
            Iterable[tuple[tuple[Any, ...], pd.DataFrame]]
            | pandas.core.groupby.generic.DataFrameGroupBy[
                tuple[Any, ...], Literal[True]
            ]
        ) = [((None,), data)]
    else:
        grouper = data.groupby(groupby)

    if progress_grouping or progress:
        if progress_grouping is None:
            progress_grouping = get_tqdm_auto(desc="Grouping data to save")

        grouper = progress_grouping(grouper)

    write_groups_l = []
    index_out_l = []
    file_map_out = pd.Series(
        [],
        index=pd.Index([], name="file_id"),
        name="file_path",
    )
    for increment, (_, df) in enumerate(grouper):
        file_id = min_file_id + increment

        new_file_path = db.get_new_data_file_path(file_id)

        file_map_out.loc[file_id] = new_file_path  # type: ignore # pandas types confused about what they support
        index_out_l.append(
            pd.DataFrame(
                np.full(len(df.index), file_id), index=df.index, columns=["file_id"]
            )
        )

        write_groups_l.append(
            SaveAction(info=df, info_kind=DBFileType.DATA, save_path=new_file_path)
        )

    if index_non_data is None:
        index_out = pd.concat(index_out_l)
    else:
        index_out = pd.concat(
            [index_non_data.reorder_levels(data.index.names), *index_out_l]
        )

    if file_map_non_data is not None:
        file_map_out = pd.concat([file_map_non_data, file_map_out])

    # Write the index first as it can be slow if very big
    write_groups_l.insert(
        0,
        SaveAction(info=index_out, info_kind=DBFileType.INDEX, save_path=db.index_file),
    )
    # Write the file map last, it is almost always cheapest
    write_groups_l.append(
        SaveAction(
            info=file_map_out, info_kind=DBFileType.FILE_MAP, save_path=db.file_map_file
        )
    )

    save_files(
        write_groups_l,
        backend=db.backend,
        parallel_op_config=parallel_op_config,
        progress=progress,
        max_workers=max_workers,
    )


def save_files(
    save_actions: Iterable[SaveAction],
    backend: OpenSCMDBBackend,
    parallel_op_config: ParallelOpConfig | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Save database information to disk

    Parameters
    ----------
    save_actions
        Iterable of save actions

    backend
        Backend to use to write the files

    parallel_op_config
        Configuration for executing the operation in parallel with progress bars

        If not supplied, we use the values of `progress` and `max_workers`.

    progress
        Should progress bar(s) be used to display the progress of the deletion?

        Only used if `parallel_op_config` is `None`.

    max_workers
        Maximum number of workers to use for parallel processing.

        If supplied, we create an instance of
        [concurrent.futures.ProcessPoolExecutor][]
        with the provided number of workers.
        A process pool seems to be the sensible default from our experimentation,
        but it is not a universally better choice.
        If you need something else because of how your database is set up,
        simply pass `parallel_op_config`
        rather than using the shortcut of passing `max_workers`.

        If not supplied, the saving is executed serially.

        Only used if `parallel_op_config` is `None`.
    """
    iterable_input: Iterable[SaveAction] | list[SaveAction] = save_actions

    # Stick the whole thing in a try finally block so we shutdown
    # the parallel pool, even if interrupted, if we created it.
    try:
        if parallel_op_config is None:
            parallel_op_config_use = ParallelOpConfig.from_user_facing(
                progress=progress,
                progress_results_kwargs=dict(desc="File saving"),
                progress_parallel_submission_kwargs=dict(
                    desc="Submitting files to the parallel executor"
                ),
                max_workers=max_workers,
                # Process pool by default as basic tests suggest
                # that writing is CPU-bound.
                # See the docs for nuance though.
                parallel_pool_cls=concurrent.futures.ProcessPoolExecutor,
            )
        else:
            parallel_op_config_use = parallel_op_config

        if parallel_op_config_use.progress_results is not None:
            # Wrap in list to force the length to be available to any progress bar.
            # This might be the wrong decision in a weird edge case,
            # but it's convenient enough that I'm willing to take that risk
            iterable_input = list(iterable_input)

        apply_op_parallel_progress(
            func_to_call=save_file,
            iterable_input=iterable_input,
            parallel_op_config=parallel_op_config_use,
            backend=backend,
        )

    finally:
        if parallel_op_config_use.executor_created_in_class_method:
            if parallel_op_config_use.executor is None:  # pragma: no cover
                # Should be impossible to get here
                raise AssertionError

            parallel_op_config_use.executor.shutdown()


def save_file(
    save_action: SaveAction,
    backend: OpenSCMDBBackend,
) -> None:
    """
    Save a file to disk

    Parameters
    ----------
    save_action
        Save action to perform

    backend
        Back-end to use for writing data
    """
    if save_action.info_kind == DBFileType.DATA:
        if isinstance(save_action.info, pd.Series):  # pragma: no cover
            # Should be impossible to get here
            raise TypeError

        backend.save_data(save_action.info, save_action.save_path)

    elif save_action.info_kind == DBFileType.INDEX:
        if isinstance(save_action.info, pd.Series):  # pragma: no cover
            # Should be impossible to get here
            raise TypeError

        backend.save_index(
            index=save_action.info,
            index_file=save_action.save_path,
        )

    elif save_action.info_kind == DBFileType.FILE_MAP:
        if isinstance(save_action.info, pd.DataFrame):  # pragma: no cover
            # Should be impossible to get here
            raise TypeError

        backend.save_file_map(
            file_map=save_action.info,
            file_map_file=save_action.save_path,
        )

    else:
        raise NotImplementedError(save_action.info_kind)


__all__ = [
    "AlreadyInDBError",
    "CSVBackend",
    "EmptyDBError",
    "FeatherBackend",
    "OpenSCMDB",
    "OpenSCMDBBackend",
    "netCDFBackend",
]
