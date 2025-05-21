"""
Definition of our key [OpenSCMDB][(m).] class
"""

from __future__ import annotations

import tarfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from attrs import define, field

from pandas_openscm.db.backends import DATA_BACKENDS, INDEX_BACKENDS
from pandas_openscm.db.deleting import delete_files
from pandas_openscm.db.interfaces import OpenSCMDBDataBackend, OpenSCMDBIndexBackend
from pandas_openscm.db.loading import (
    load_data,
    load_db_file_map,
    load_db_index,
    load_db_metadata,
)
from pandas_openscm.db.path_handling import DBPath
from pandas_openscm.db.reader import OpenSCMDBReader
from pandas_openscm.db.rewriting import make_move_plan, rewrite_files
from pandas_openscm.db.saving import save_data
from pandas_openscm.exceptions import MissingOptionalDependencyError
from pandas_openscm.index_manipulation import unify_index_levels_check_index_types
from pandas_openscm.indexing import multi_index_match
from pandas_openscm.parallelisation import (
    ParallelOpConfig,
    ProgressLike,
)

if TYPE_CHECKING:
    import filelock
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
        try:
            import filelock
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "default_index_file_lock", requirement="filelock"
            ) from exc

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

            If not supplied, we use [self.index_file_lock][(c)].

        Returns
        -------
        :
            Database reader
        """
        if isinstance(lock, bool):
            if lock:
                try:
                    import filelock
                except ImportError as exc:
                    raise MissingOptionalDependencyError(  # noqa: TRY003
                        "create_reader(..., lock=True, ...)", requirement="filelock"
                    ) from exc

                # Create a new lock for the reader
                lock = filelock.FileLock(self.index_file_lock_path)

            else:
                # Convert to None
                lock = None

        db_index = self.load_index(index_file_lock=index_file_lock)
        db_file_map = self.load_file_map(index_file_lock=index_file_lock)

        res = OpenSCMDBReader(
            backend_data=self.backend_data,
            db_dir=self.db_dir,
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

            If not supplied, we use [self.index_file_lock][(c)].

        parallel_op_config
            Configuration for executing the operation in parallel with progress bars

            If not supplied, we use the values of `progress` and `max_workers`.

        progress
            Should progress bar(s) be used to display the progress of the deletion?

            Only used if `parallel_op_config` is `None`.

        max_workers
            Maximum number of workers to use for parallel processing.

            If supplied, we create an instance of
            [concurrent.futures.ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor)
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

    @classmethod
    def from_gzipped_tar_archive(
        cls,
        tar_archive: Path,
        db_dir: Path,
        backend_data: OpenSCMDBDataBackend | None = None,
        backend_index: OpenSCMDBIndexBackend | None = None,
    ) -> OpenSCMDB:
        """
        Initialise from a gzipped tar archive

        This also unpacks the files to disk

        Parameters
        ----------
        tar_archive
            Tar archive from which to initialise

        db_dir
            Directory in which to unpack the database

        backend_data
            Backend to use for handling the data

        backend_index
            Backend to use for handling the index

        Returns
        -------
        :
            Initialised database
        """
        with tarfile.open(tar_archive, "r") as tar:
            for member in tar.getmembers():
                if not member.isreg():
                    # Only extract files
                    continue
                # Extract to the db_dir
                member.name = Path(member.name).name
                tar.extract(member, db_dir)
                if backend_index is None and member.name.startswith("index"):
                    backend_index = INDEX_BACKENDS.guess_backend(member.name)

                if backend_data is None and not any(
                    member.name.startswith(v) for v in ["index", "filemap"]
                ):
                    backend_data = DATA_BACKENDS.guess_backend(member.name)

        if backend_data is None:  # pragma: no cover
            # Should be impossible to get here
            raise TypeError(backend_data)

        if backend_index is None:  # pragma: no cover
            # Should be impossible to get here
            raise TypeError(backend_index)

        res = cls(backend_data=backend_data, backend_index=backend_index, db_dir=db_dir)

        return res

    def get_new_data_file_path(self, file_id: int) -> DBPath:
        """
        Get the path in which to write a new data file

        Parameters
        ----------
        file_id
            ID to associate with the file

        Returns
        -------
        :
            Information about the path in which to write the new data

        Raises
        ------
        FileExistsError
            A file already exists for the given `file_id`
        """
        file_path = self.db_dir / f"{file_id}{self.backend_data.ext}"

        if file_path.exists():
            raise FileExistsError(file_path)

        return DBPath.from_abs_path_and_db_dir(abs=file_path, db_dir=self.db_dir)

    def load(  # noqa: PLR0913
        self,
        selector: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector | None = None,
        *,
        index_file_lock: filelock.BaseFileLock | None = None,
        out_columns_type: type | None = None,
        out_columns_name: str | None = None,
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

            If not supplied, we use [self.index_file_lock][(c)].

        out_columns_type
            Type to set the output columns to.

            If not supplied, we don't set the output columns' type.

        out_columns_name
            The name for the columns in the output.

            If not supplied, we don't set the output columns' name.

            This can also be set with
            [pd.DataFrame.rename_axis][pandas.DataFrame.rename_axis]
            but we provide it here for convenience
            (and in case you couldn't find this trick for ages, like us).

        parallel_op_config
            Configuration for executing the operation in parallel with progress bars

            If not supplied, we use the values of `progress` and `max_workers`.

        progress
            Should progress bar(s) be used to display the progress of the deletion?

            Only used if `parallel_op_config` is `None`.

        max_workers
            Maximum number of workers to use for parallel processing.

            If supplied, we create an instance of
            [concurrent.futures.ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor)
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
                db_dir=self.db_dir,
                selector=selector,
                out_columns_type=out_columns_type,
                out_columns_name=out_columns_name,
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

            If not supplied, we use [self.index_file_lock][(c)].

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

            If not supplied, we use [self.index_file_lock][(c)].

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

            If not supplied, we use [self.index_file_lock][(c)].

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

            If not supplied, we use [self.index_file_lock][(c)].

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
                move_plan = None
                index_non_data = None
                file_map_non_data = None
                min_file_id = 0

            else:
                file_map_db = self.load_file_map(index_file_lock=index_file_lock)
                index_db = self.load_index(index_file_lock=index_file_lock)
                if not allow_overwrite:
                    data_index_unified, index_db_index_unified = (
                        unify_index_levels_check_index_types(data.index, index_db.index)
                    )
                    overwrite_required = multi_index_match(
                        data_index_unified, index_db_index_unified
                    )

                    if overwrite_required.any():
                        data_to_write_already_in_db = data.loc[overwrite_required, :]
                        raise AlreadyInDBError(
                            already_in_db=data_to_write_already_in_db
                        )

                move_plan = make_move_plan(
                    index_start=index_db,
                    file_map_start=file_map_db,
                    data_to_write=data,
                    get_new_data_file_path=self.get_new_data_file_path,
                    db_dir=self.db_dir,
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

                index_non_data = move_plan.moved_index
                file_map_non_data = move_plan.moved_file_map
                min_file_id = current_largest_file_id + 1

            save_data(
                data,
                backend_data=self.backend_data,
                get_new_data_file_path=self.get_new_data_file_path,
                backend_index=self.backend_index,
                index_file=self.index_file,
                file_map_file=self.file_map_file,
                index_non_data=index_non_data,
                file_map_non_data=file_map_non_data,
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
            if move_plan is not None and move_plan.delete_paths is not None:
                delete_files(
                    files_to_delete=move_plan.delete_paths,
                    parallel_op_config=parallel_op_config_delete,
                    progress=progress,
                    max_workers=max_workers,
                )

    def to_gzipped_tar_archive(self, out_file: Path, mode: str = "w:gz") -> Path:
        """
        Convert to a gzipped tar archive

        Parameters
        ----------
        out_file
            File in which to write the output

        mode
            Mode to use to open `out_file`

        Returns
        -------
        :
            Path to the gzipped tar archive

            This is the same as `out_file`, but is returned for convenience.
        """
        with tarfile.open(out_file, mode=mode) as tar:
            tar.add(self.db_dir, arcname="db")

        return out_file
