"""
Functionality for re-writing a database

Mostly used to make way for new data to be written
or to overwrite old data
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from attrs import define

from pandas_openscm.db.interfaces import OpenSCMDBDataBackend
from pandas_openscm.db.path_handling import DBPath
from pandas_openscm.index_manipulation import (
    unify_index_levels_check_index_types,
    update_index_from_candidates,
)
from pandas_openscm.indexing import mi_loc, multi_index_match
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

if TYPE_CHECKING:
    import pandas.core.indexes.frozen


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


@define
class ReWriteAction:
    """Description of a re-write action"""

    from_file: Path
    """File from which to load the data"""

    to_file: Path
    """File in which to write the re-written data"""

    locator: pd.MultiIndex
    """Locator which specifies which data to re-write"""


def rewrite_file(
    rewrite_action: ReWriteAction,
    backend: OpenSCMDBDataBackend,
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
    data_all = backend.load_data(rewrite_action.from_file)
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
    backend: OpenSCMDBDataBackend,
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
        [concurrent.futures.ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor)
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


def make_move_plan(
    *,
    index_start: pd.DataFrame,
    file_map_start: pd.Series[Path],  # type: ignore # pandas confused about ability to support Path
    data_to_write: pd.DataFrame,
    get_new_data_file_path: Callable[[int], DBPath],
    db_dir: Path,
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

    get_new_data_file_path
        Callable which, given an integer, returns the path info for the new data file

    db_dir
        Database directory

    Returns
    -------
    :
        Plan for moving data to make room for the new data
    """
    index_start_index_unified, data_to_write_index_unified = (
        unify_index_levels_check_index_types(index_start.index, data_to_write.index)
    )
    in_data_to_write = pd.Series(
        multi_index_match(index_start_index_unified, data_to_write_index_unified),  # type: ignore # pandas type hints confused
        index=index_start.set_index("file_id", append=True).index,
    )

    grouper = in_data_to_write.groupby("file_id")
    no_overwrite = ~grouper.apply(np.any)
    if no_overwrite.all():
        # Don't need to move anything, just return what we started with
        return MovePlan(
            moved_index=index_start,
            moved_file_map=file_map_start,
            rewrite_actions=None,
            delete_paths=None,
        )

    full_overwrite: pd.Series[bool] = grouper.apply(np.all)
    partial_overwrite = ~(full_overwrite | no_overwrite)
    if not partial_overwrite.any():
        # Don't need to move anything,
        # but do no need to delete some files
        # to make way for the parts of the index that will be overwritten
        # (would be even more efficient to just update the file IDs,
        # but that would create a coupling I can't get my head around right now).
        delete_file_ids = full_overwrite.index[full_overwrite]
        delete_paths = (db_dir / v for v in file_map_start.loc[delete_file_ids])
        moved_index = index_start[~index_start["file_id"].isin(delete_file_ids)]
        file_map_out = file_map_start.loc[moved_index["file_id"].unique()]

        return MovePlan(
            moved_index=moved_index,
            moved_file_map=file_map_out,
            rewrite_actions=None,
            delete_paths=tuple(delete_paths),
        )

    # Neither nothing to do or only deletions i.e. the fun part.
    to_keep_via_rewrite = partial_overwrite & ~in_data_to_write

    full_overwrite_file_ids = full_overwrite.index[full_overwrite]
    partial_overwrite_file_ids = partial_overwrite.index[partial_overwrite]
    file_ids_to_delete = np.union1d(full_overwrite_file_ids, partial_overwrite_file_ids)
    delete_paths = (db_dir / v for v in file_map_start.loc[file_ids_to_delete])

    file_id_map = {}
    max_file_id_start = file_map_start.index.max()
    # Start just with the files that aren't affected by the overwrite
    file_map_out = file_map_start[no_overwrite].copy()
    rewrite_actions_l = []
    for increment, (file_id_old, fiddf) in enumerate(
        # Figure out where to rewrite the data that needs to be rewritten
        to_keep_via_rewrite.loc[to_keep_via_rewrite].groupby("file_id")
    ):
        new_file_id = max_file_id_start + 1 + increment

        new_db_path = get_new_data_file_path(new_file_id)
        file_map_out.loc[new_file_id] = new_db_path.rel_db

        rewrite_actions_l.append(
            ReWriteAction(
                from_file=db_dir / file_map_start.loc[file_id_old],
                to_file=new_db_path.abs,
                locator=fiddf.index.droplevel("file_id"),
            )
        )
        file_id_map[file_id_old] = new_file_id

    index_keep_via_rewrite = in_data_to_write[
        ~in_data_to_write
        & in_data_to_write.index.get_level_values("file_id").isin(
            partial_overwrite_file_ids
        )
    ].reset_index("file_id")[["file_id"]]

    index_keep_via_rewrite["file_id"] = index_keep_via_rewrite["file_id"].map(
        file_id_map
    )
    if index_keep_via_rewrite["file_id"].isnull().any():  # pragma: no cover
        # Something has gone wrong, everything should be remapped somewhere
        raise AssertionError

    moved_index = pd.concat(
        [
            # Bits of the index which won't be overwritten
            index_start[~index_start["file_id"].isin(file_ids_to_delete)],
            # Bits of the index which are being kept after a rewrite
            index_keep_via_rewrite,
        ]
    )
    res = MovePlan(
        moved_index=moved_index,
        moved_file_map=file_map_out,
        rewrite_actions=tuple(rewrite_actions_l),
        delete_paths=tuple(delete_paths),
    )

    return res
