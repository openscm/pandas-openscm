"""
Functionality for saving data
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from attrs import define

from pandas_openscm.db.interfaces import OpenSCMDBDataBackend, OpenSCMDBIndexBackend
from pandas_openscm.db.path_handling import DBPath
from pandas_openscm.index_manipulation import (
    unify_index_levels_check_index_types,
)
from pandas_openscm.parallelisation import (
    ParallelOpConfig,
    ProgressLike,
    apply_op_parallel_progress,
    get_tqdm_auto,
)

if TYPE_CHECKING:
    import pandas.core.groupby.generic
    import pandas.core.indexes.frozen


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


def save_data(  # noqa: PLR0913
    data: pd.DataFrame,
    *,
    backend_data: OpenSCMDBDataBackend,
    get_new_data_file_path: Callable[[int], DBPath],
    backend_index: OpenSCMDBIndexBackend,
    index_file: Path,
    file_map_file: Path,
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

    backend_data
        Backend to use to save the data

    get_new_data_file_path
        Callable which, given an integer, returns the path info for the new data file

    backend_index
        Backend to use to save the index

    index_file
        File in which to save the index

    file_map_file
        File in which to save the file map

    index_non_data
        Index that is already in the database but isn't related to data.

        If supplied, this is combined with the index generated for `data`
        before we write the database's index.

    file_map_non_data
        File map that is already in the database but isn't related to `data`.

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
        [concurrent.futures.ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor)
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
                # Switch to the below
                # when we switch mypy checks to using python > 3.9
                # tuple[Any, ...], Literal[True]
                tuple[Any, ...]
            ]
        ) = [((None,), data)]
    else:
        # Only want combos that are actually in the data
        grouper = data.groupby(groupby, observed=True)

    if progress_grouping or progress:
        if progress_grouping is None:
            progress_grouping = get_tqdm_auto(desc="Grouping data to save")

        grouper = progress_grouping(grouper)

    if index_non_data is None:
        index_non_data_unified_index = None
    else:
        unified_index = unify_index_levels_check_index_types(
            index_non_data.index, data.index[:1]
        )[0]
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

        new_db_path = get_new_data_file_path(file_id)

        file_map_out.loc[file_id] = new_db_path.rel_db  # type: ignore # pandas types confused about what they support
        if index_non_data_unified_index is None:
            df_index_unified = df.index
        else:
            _, df_index_unified = unify_index_levels_check_index_types(
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
                backend=backend_data,
                save_path=new_db_path.abs,
            )
        )

    if index_non_data_unified_index is None:
        index_out = pd.concat(index_data_out_l)
    else:
        index_out = pd.concat([index_non_data_unified_index, *index_data_out_l])

    if file_map_non_data is not None:
        file_map_out = pd.concat([file_map_non_data, file_map_out])  # type: ignore # pandas-stubs confused

    # Write the index first as it can be slow if very big
    write_groups_l.insert(
        0,
        SaveAction(
            info=index_out,
            info_kind=DBFileType.INDEX,
            backend=backend_index,
            save_path=index_file,
        ),
    )
    # Write the file map last, it is almost always cheapest
    write_groups_l.append(
        SaveAction(
            info=file_map_out,
            info_kind=DBFileType.FILE_MAP,
            backend=backend_index,
            save_path=file_map_file,
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
        [concurrent.futures.ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor)
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
