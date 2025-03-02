"""
Functionality for deleting data
"""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Iterable
from pathlib import Path

from pandas_openscm.parallelisation import (
    ParallelOpConfig,
    apply_op_parallel_progress,
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
        [concurrent.futures.ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor)
        with the provided number of workers
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
