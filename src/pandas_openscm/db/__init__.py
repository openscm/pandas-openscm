"""
Database
"""
# TODO: split into:
# deleting.py
# rewriting.py/moving.py
# - move plan stuff in here too
# saving.py

from __future__ import annotations

import concurrent.futures
import os
import warnings
from collections.abc import Iterable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import filelock
import numpy as np
import pandas as pd
from attrs import define, field

from pandas_openscm.db.csv import CSVDataBackend, CSVIndexBackend
from pandas_openscm.db.feather import FeatherDataBackend, FeatherIndexBackend
from pandas_openscm.db.in_memory import InMemoryDataBackend, InMemoryIndexBackend
from pandas_openscm.db.interfaces import OpenSCMDBDataBackend, OpenSCMDBIndexBackend
from pandas_openscm.db.loading import (
    load_data,
    load_db_file_map,
    load_db_index,
    load_db_metadata,
)
from pandas_openscm.db.netcdf import netCDFDataBackend, netCDFIndexBackend
from pandas_openscm.db.reader import OpenSCMDBReader
from pandas_openscm.db.rewriting import make_move_plan, rewrite_files
from pandas_openscm.index_manipulation import unify_index_levels
from pandas_openscm.indexing import multi_index_match
from pandas_openscm.parallelisation import (
    ParallelOpConfig,
    ProgressLike,
    apply_op_parallel_progress,
    get_tqdm_auto,
)

if TYPE_CHECKING:
    import pandas.core.groupby.generic
    import pandas.core.indexes.frozen
    import pandas_indexing as pix


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

    backend: OpenSCMDBDataBackend | OpenSCMDBIndexBackend
    """Backend to use to save the data to disk"""

    save_path: Path
    """Path in which to save the information"""


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

    backend_data: OpenSCMDBDataBackend = field(kw_only=True)
    """
    The backend for (de-)serialising data (from) to disk
    """

    backend_index: OpenSCMDBIndexBackend = field(kw_only=True)
    """
    The backend for (de-)serialising the database index (from) to disk
    """

    db_dir: Path = field(kw_only=True)
    """
    Path in which the database is stored

    Both the index and the data files will be written in this directory.
    """

    index_file_lock: filelock.BaseFileLock = field(kw_only=True)
    """
    Lock for the index file
    """
    # Note to devs: filelock releases the lock when __del__ is called
    # (i.e. when the lock instance is garbage collected).
    # Hence, you have to keep a reference to this around
    # if you want it to do anything.
    # For a while, we made this a property that created the lock when requested.
    # That was super confusing as, if the reference to the created lock wasn't kept,
    # the lock would immediately be released.

    @index_file_lock.default
    def default_index_file_lock(self) -> filelock.BaseFileLock:
        """Get default lock for the back-end's index file"""
        return filelock.FileLock(self.index_file_lock_path)

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
        return self.db_dir / f"filemap{self.backend_index.ext}"

    @property
    def index_file(self) -> Path:
        """
        The file in which the database's index is stored

        Returns
        -------
        :
            Path to the index file
        """
        return self.db_dir / f"index{self.backend_index.ext}"

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

    def create_reader(
        self,
        *,
        lock: bool | filelock.BaseFileLock | None = True,
        index_file_lock: filelock.BaseFileLock | None = None,
    ) -> OpenSCMDBReader:
        """
        Create a database reader

        Parameters
        ----------
        lock
            Lock to give to the reader.

            If `True`, we create a new lock for the database, such that,
            if the reader is holding the lock,
            no operations can be performed on the database.

            If `False`, the reader is not given any lock.

        index_file_lock
            Lock for the database's index file

            Used while loading the index from disk.

            If not supplied, we use [`self.index_file_lock`][(c)].

        Returns
        -------
        :
            Database reader
        """
        if isinstance(lock, bool):
            if lock:
                # Create a new lock for the reader
                lock = filelock.FileLock(self.index_file_lock_path)
            else:
                # Convert to None
                lock = None

        db_index = self.load_index(index_file_lock=index_file_lock)
        db_file_map = self.load_file_map(index_file_lock=index_file_lock)

        res = OpenSCMDBReader(
            backend_data=self.backend_data,
            db_index=db_index,
            db_file_map=db_file_map,
            lock=lock,
        )

        return res

    def delete(
        self,
        *,
        index_file_lock: filelock.BaseFileLock | None = None,
        parallel_op_config: ParallelOpConfig | None = None,
        progress: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """
        Delete all data in the database

        Parameters
        ----------
        index_file_lock
            Lock for the database's index file

            If not supplied, we use [`self.index_file_lock`][(c)].

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
        if index_file_lock is None:
            index_file_lock = self.index_file_lock

        with index_file_lock:
            files_to_delete = {
                *self.db_dir.glob(f"*{self.backend_data.ext}"),
                *self.db_dir.glob(f"*{self.backend_index.ext}"),
            }
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
        file_path = self.db_dir / f"{file_id}{self.backend_data.ext}"

        if file_path.exists():
            raise FileExistsError(file_path)

        return file_path

    def load(  # noqa: PLR0913
        self,
        selector: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector | None = None,
        *,
        index_file_lock: filelock.BaseFileLock | None = None,
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

        index_file_lock
            Lock for the database's index file

            If not supplied, we use [`self.index_file_lock`][(c)].

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

        Returns
        -------
        :
            Loaded data

        Raises
        ------
        EmptyDBError
            The database is empty
        """
        if self.is_empty:
            raise EmptyDBError(self)

        if index_file_lock is None:
            index_file_lock = self.index_file_lock

        with index_file_lock:
            file_map = self.load_file_map(index_file_lock=index_file_lock)
            index = self.load_index(index_file_lock=index_file_lock)

            res = load_data(
                backend_data=self.backend_data,
                db_index=index,
                db_file_map=file_map,
                selector=selector,
                out_columns_type=out_columns_type,
                parallel_op_config=parallel_op_config,
                progress=progress,
                max_workers=max_workers,
            )

        return res

    def load_file_map(
        self,
        *,
        index_file_lock: filelock.BaseFileLock | None = None,
    ) -> pd.Series[Path]:  # type: ignore # pandas type hints confused about what they support
        """
        Load the file map

        Parameters
        ----------
        index_file_lock
            Lock for the database's index file

            If not supplied, we use [`self.index_file_lock`][(c)].

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

        if index_file_lock is None:
            index_file_lock = self.index_file_lock

        with index_file_lock:
            file_map = load_db_file_map(
                backend_index=self.backend_index, file_map_file=self.file_map_file
            )

        return file_map

    def load_index(
        self,
        *,
        index_file_lock: filelock.BaseFileLock | None = None,
    ) -> pd.DataFrame:
        """
        Load the index

        Parameters
        ----------
        index_file_lock
            Lock for the database's index file

            If not supplied, we use [`self.index_file_lock`][(c)].

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

        if index_file_lock is None:
            index_file_lock = self.index_file_lock

        with index_file_lock:
            index = load_db_index(
                backend_index=self.backend_index,
                index_file=self.index_file,
            )

        return index

    def load_metadata(
        self,
        *,
        index_file_lock: filelock.BaseFileLock | None = None,
    ) -> pd.MultiIndex:
        """
        Load the database's metadata

        Parameters
        ----------
        index_file_lock
            Lock for the database's index file

            If not supplied, we use [`self.index_file_lock`][(c)].

        Returns
        -------
        :
            Loaded metadata
        """
        if not self.index_file.exists():
            raise EmptyDBError(self)

        if index_file_lock is None:
            index_file_lock = self.index_file_lock

        with index_file_lock:
            metadata = load_db_metadata(
                backend_index=self.backend_index, index_file=self.index_file
            )

        return metadata

    def save(  # noqa: PLR0913
        self,
        data: pd.DataFrame,
        *,
        index_file_lock: filelock.BaseFileLock | None = None,
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

        Parameters
        ----------
        data
            Data to add to the database

        index_file_lock
            Lock for the database's index file

            If not supplied, we use [`self.index_file_lock`][(c)].

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
            msg = (
                "`data.index` must be an instance of `pd.MultiIndex`. "
                f"Received {type(data.index)=}"
            )
            raise TypeError(msg)

        if data.index.duplicated().any():
            duplicate_rows = data.index.duplicated(keep=False)
            duplicates = data.loc[duplicate_rows, :]
            msg = (
                "`data` contains rows with the same metadata. "
                f"duplicates=\n{duplicates}"
            )

            raise ValueError(msg)

        if index_file_lock is None:
            index_file_lock = self.index_file_lock

        with index_file_lock:
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

            file_map_db = self.load_file_map(index_file_lock=index_file_lock)
            index_db = self.load_index(index_file_lock=index_file_lock)
            if not allow_overwrite:
                if not isinstance(index_db.index, pd.MultiIndex):  # pragma: no cover
                    # Should be impossible to get here
                    raise TypeError

                data_index_unified, index_db_index_unified = unify_index_levels(
                    data.index, index_db.index
                )
                overwrite_required = multi_index_match(
                    data_index_unified, index_db_index_unified
                )

                if overwrite_required.any():
                    data_to_write_already_in_db = data.loc[overwrite_required, :]
                    raise AlreadyInDBError(already_in_db=data_to_write_already_in_db)

            move_plan = make_move_plan(
                index_start=index_db,
                file_map_start=file_map_db,
                data_to_write=data,
                get_new_data_file_path=self.get_new_data_file_path,
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
                    backend=self.backend_data,
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

    if index_non_data is None:
        index_non_data_unified_index = None
    else:
        unified_index = unify_index_levels(index_non_data.index, data.index[:1])[0]
        index_non_data_unified_index = pd.DataFrame(
            index_non_data.values,
            index=unified_index,
            columns=index_non_data.columns,
        )

    write_groups_l = []
    index_data_out_l = []
    file_map_out = pd.Series(
        [],
        index=pd.Index([], name="file_id"),
        name="file_path",
    )
    for increment, (_, df) in enumerate(grouper):
        file_id = min_file_id + increment

        new_file_path = db.get_new_data_file_path(file_id)

        file_map_out.loc[file_id] = new_file_path  # type: ignore # pandas types confused about what they support
        if index_non_data_unified_index is None:
            df_index_unified = df.index
        else:
            _, df_index_unified = unify_index_levels(
                index_non_data_unified_index.index[:1], df.index
            )

        index_data_out_l.append(
            pd.DataFrame(
                np.full(df.index.shape[0], file_id),
                index=df_index_unified,
                columns=["file_id"],
            )
        )

        write_groups_l.append(
            SaveAction(
                info=df,
                info_kind=DBFileType.DATA,
                backend=db.backend_data,
                save_path=new_file_path,
            )
        )

    if index_non_data_unified_index is None:
        index_out = pd.concat(index_data_out_l)
    else:
        index_out = pd.concat([index_non_data_unified_index, *index_data_out_l])

    if file_map_non_data is not None:
        file_map_out = pd.concat([file_map_non_data, file_map_out])

    # Write the index first as it can be slow if very big
    write_groups_l.insert(
        0,
        SaveAction(
            info=index_out,
            info_kind=DBFileType.INDEX,
            backend=db.backend_index,
            save_path=db.index_file,
        ),
    )
    # Write the file map last, it is almost always cheapest
    write_groups_l.append(
        SaveAction(
            info=file_map_out,
            info_kind=DBFileType.FILE_MAP,
            backend=db.backend_index,
            save_path=db.file_map_file,
        )
    )

    save_files(
        write_groups_l,
        parallel_op_config=parallel_op_config,
        progress=progress,
        max_workers=max_workers,
    )


def save_files(
    save_actions: Iterable[SaveAction],
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
        )

    finally:
        if parallel_op_config_use.executor_created_in_class_method:
            if parallel_op_config_use.executor is None:  # pragma: no cover
                # Should be impossible to get here
                raise AssertionError

            parallel_op_config_use.executor.shutdown()


def save_file(save_action: SaveAction) -> None:
    """
    Save a file to disk

    Parameters
    ----------
    save_action
        Save action to perform
    """
    if save_action.info_kind == DBFileType.DATA:
        if isinstance(save_action.info, pd.Series) or isinstance(
            save_action.backend, OpenSCMDBIndexBackend
        ):  # pragma: no cover
            # Should be impossible to get here
            raise TypeError

        save_action.backend.save_data(save_action.info, save_action.save_path)

    elif save_action.info_kind == DBFileType.INDEX:
        if isinstance(save_action.info, pd.Series) or isinstance(
            save_action.backend, OpenSCMDBDataBackend
        ):  # pragma: no cover
            # Should be impossible to get here
            raise TypeError

        save_action.backend.save_index(
            index=save_action.info,
            index_file=save_action.save_path,
        )

    elif save_action.info_kind == DBFileType.FILE_MAP:
        if isinstance(save_action.info, pd.DataFrame) or isinstance(
            save_action.backend, OpenSCMDBDataBackend
        ):  # pragma: no cover
            # Should be impossible to get here
            raise TypeError

        save_action.backend.save_file_map(
            file_map=save_action.info,
            file_map_file=save_action.save_path,
        )

    else:
        raise NotImplementedError(save_action.info_kind)


__all__ = [
    "AlreadyInDBError",
    "CSVDataBackend",
    "CSVIndexBackend",
    "EmptyDBError",
    "FeatherDataBackend",
    "FeatherIndexBackend",
    "InMemoryDataBackend",
    "InMemoryIndexBackend",
    "OpenSCMDB",
    "OpenSCMDBBackend",
    "OpenSCMDBIndexBackend",
    "netCDFDataBackend",
    "netCDFIndexBackend",
]
