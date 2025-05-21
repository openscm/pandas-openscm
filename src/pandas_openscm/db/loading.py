"""
Loading of data from disk
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pandas_openscm.db.interfaces import OpenSCMDBDataBackend, OpenSCMDBIndexBackend
from pandas_openscm.index_manipulation import (
    unify_index_levels_check_index_types,
    update_index_from_candidates,
)
from pandas_openscm.indexing import mi_loc
from pandas_openscm.parallelisation import (
    ParallelOpConfig,
    apply_op_parallel_progress,
)

if TYPE_CHECKING:
    import pandas.core.groupby.generic
    import pandas.core.indexes.frozen
    import pandas_indexing as pix


def load_data(  # noqa: PLR0913
    *,
    backend_data: OpenSCMDBDataBackend,
    db_index: pd.DataFrame,
    db_file_map: pd.Series[Path],  # type: ignore # pandas type hints confused about what they support
    db_dir: Path,
    selector: pd.Index[Any] | pd.MultiIndex | pix.selectors.Selector | None = None,
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
    backend_data
        Backend to use to load data from disk

    db_index
        Index of the database from which to load

    db_file_map
        File map of the database from which to load

    db_dir
        The directory in which the database lives

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
    if selector is None:
        index_to_load = db_index
    else:
        index_to_load = mi_loc(db_index, selector)

    files_to_load = (db_dir / v for v in db_file_map[index_to_load["file_id"].unique()])
    loaded_l = load_data_files(
        files_to_load=files_to_load,
        backend_data=backend_data,
        parallel_op_config=parallel_op_config,
        progress=progress,
        max_workers=max_workers,
    )

    if backend_data.preserves_index and any(
        v.index.names != loaded_l[0].index.names for v in loaded_l
    ):
        base_idx = index_to_load.index[:1]
        for i in range(len(loaded_l)):
            new_index = unify_index_levels_check_index_types(
                base_idx, loaded_l[i].index
            )[1]
            loaded_l[i].index = new_index

    res = pd.concat(loaded_l)

    if not backend_data.preserves_index:
        index_names: pandas.core.indexes.frozen.FrozenList = index_to_load.index.names  # type: ignore # pandas type hints wrong
        res = update_index_from_candidates(res, index_names.difference({"file_id"}))

    # Look up only the indexes we want
    # just in case the data we loaded had more than we asked for
    # (because the files aren't saved with exactly the right granularity
    # for the query that has been requested).
    if selector is not None:
        res = mi_loc(res, selector)

    if out_columns_type is not None:
        res.columns = res.columns.astype(out_columns_type)

    if out_columns_name is not None:
        res = res.rename_axis(out_columns_name, axis="columns")

    return res


def load_data_files(
    *,
    files_to_load: Iterable[Path],
    backend_data: OpenSCMDBDataBackend,
    parallel_op_config: ParallelOpConfig | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, ...]:
    """
    Load a number of data files

    Parameters
    ----------
    files_to_load
        Files to load

    backend_data
        Data backend to use to load the files

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
        Loaded data from each file

    See Also
    --------
    [load_data][(m).], which loads data while applying selectors
    (in contrast to this function which just loads all data).
    """
    iterable_input: Iterable[Path] | list[Path] = files_to_load

    # Stick the whole thing in a try finally block so we shutdown
    # if we created the parallel pool, it is shut down even if interrupted.
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
            func_to_call=backend_data.load_data,
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


def load_db_index(
    *,
    backend_index: OpenSCMDBIndexBackend,
    index_file: Path,
) -> pd.DataFrame:
    """
    Load database index from file

    Parameters
    ----------
    backend_index
        Backend to use to load the index

    index_file
        File from which to load the index

    Returns
    -------
    :
        Loaded index
    """
    db_index = backend_index.load_index(index_file)

    if not backend_index.preserves_index:
        db_index = db_index.set_index(
            db_index.columns.difference(["file_id"]).to_list()
        )

    return db_index


def load_db_file_map(
    *,
    backend_index: OpenSCMDBIndexBackend,
    file_map_file: Path,
) -> pd.Series[Path]:  # type: ignore # pandas type hints confused about what they support
    """
    Load database file map from file

    Parameters
    ----------
    backend_index
        Backend to use to load the index

    file_map_file
        File from which to load the file map

    Returns
    -------
    :
        Loaded file map
    """
    file_map_raw = backend_index.load_file_map(file_map_file)
    if not backend_index.preserves_index:
        file_map_indexed = file_map_raw.set_index("file_id")
    else:
        file_map_indexed = file_map_raw

    file_map = file_map_indexed["file_path"]

    return file_map


def convert_db_index_to_metadata(db_index: pd.DataFrame) -> pd.MultiIndex:
    """
    Convert a database index to metadata

    Parameters
    ----------
    db_index
        Database index

    Returns
    -------
    :
        Metadata
    """
    if not isinstance(db_index.index, pd.MultiIndex):  # pragma: no cover
        # Should be impossible to get here
        raise TypeError(db_index.index)

    res: pd.MultiIndex = db_index.index

    return res


def load_db_metadata(
    *,
    backend_index: OpenSCMDBIndexBackend,
    index_file: Path,
) -> pd.MultiIndex:
    """
    Load database metadata from file

    Parameters
    ----------
    backend_index
        Backend to use to load the index

    index_file
        File from which to load the index (from which the metadata is inferred)

    Returns
    -------
    :
        Loaded metadata
    """
    db_index = load_db_index(backend_index=backend_index, index_file=index_file)

    return convert_db_index_to_metadata(db_index)
