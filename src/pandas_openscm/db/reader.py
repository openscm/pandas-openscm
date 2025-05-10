"""
Database reader

A small optimisation to allow for a reader that holds the index in memory,
rather than loading it from disk on every operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from attrs import define, field

from pandas_openscm.db.interfaces import OpenSCMDBDataBackend
from pandas_openscm.db.loading import convert_db_index_to_metadata, load_data
from pandas_openscm.parallelisation import (
    ParallelOpConfig,
)

if TYPE_CHECKING:
    from types import TracebackType

    import filelock
    import pandas_indexing as pix


@define
class OpenSCMDBReader:
    """
    Reader for reading data out of a database created with `OpenSCMDB`

    Holds the database file map and index in memory,
    which can make repeated read operations faster
    than using an `OpenSCMDB` instance.
    """

    backend_data: OpenSCMDBDataBackend = field(kw_only=True)
    """
    The backend for reading data from disk
    """

    db_dir: Path = field(kw_only=True)
    """
    The directory in which the database lives
    """

    db_file_map: pd.Series[Path] = field(kw_only=True)  # type: ignore # pandas type hints confused about what they support
    """
    The file map of the database from which we are reading.
    """

    db_index: pd.DataFrame = field(kw_only=True)
    """
    The index of the database from which we are reading.
    """

    lock: filelock.BaseFileLock | None = field(kw_only=True)
    """
    Lock for the database from which data is being read

    If `None`, we don't hold the lock and automatic locking is not enabled.
    """

    def __enter__(self) -> OpenSCMDBReader:
        """
        If the reader has a lock, acquire it
        """
        if self.lock is not None:
            self.lock.acquire()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        If the reader has a lock, release it
        """
        if self.lock is not None:
            self.lock.release()

    @property
    def metadata(self) -> pd.MultiIndex:
        """
        Database's metadata
        """
        return convert_db_index_to_metadata(db_index=self.db_index)

    def load(  # noqa: PLR0913
        self,
        selector: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector | None = None,
        *,
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
        """
        return load_data(
            backend_data=self.backend_data,
            db_index=self.db_index,
            db_file_map=self.db_file_map,
            db_dir=self.db_dir,
            selector=selector,
            out_columns_type=out_columns_type,
            out_columns_name=out_columns_name,
            parallel_op_config=parallel_op_config,
            progress=progress,
            max_workers=max_workers,
        )
