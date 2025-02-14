"""
Parallelisation helpers

Parallelisation is notoriously difficult.
Here we capture some helpers that we have found useful.
If you want to get started with parallelisation,
have a look at some of these resources:

- https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
- https://pythonspeed.com/articles/python-multiprocessing/
- https://joblib.readthedocs.io/en/latest/parallel.html

If you want to understand why I've gone with
[concurrent.futures][] as the default,
have a look at
[this excellent Stack Overflow answer](https://stackoverflow.com/a/60644649)
(tl;dr - [concurrent.futures][] does enough,
the added complexity of the lower-level interfaces isn't worth it).
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, Callable, Protocol, TypeVar

from attrs import define
from typing_extensions import Concatenate, ParamSpec

from pandas_openscm.exceptions import MissingOptionalDependencyError

P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")
V = TypeVar("V")


class ProgressLike(Protocol):
    """A callable that acts like something which creates a progress bar"""

    def __call__(
        self, iterable: Iterable[V], total: int | float | None = None
    ) -> Iterator[V]:
        """
        Create the progress bar

        Parameters
        ----------
        iterable
            Iterable to wrap

        total
            Total number of iterations if known.
        """


def get_tqdm_auto(**kwargs: Any) -> ProgressLike:
    """
    Get a progress bar from [tqdm.auto](https://tqdm.github.io/docs/shortcuts/#tqdmauto).

    Strictly speaking, we use `tqdm.auto.tqdm`.

    Parameters
    ----------
    **kwargs
        Passed to `tqdm.auto.tqdm`.

    Returns
    -------
    :
        Default progress bar.

    Notes
    -----
    This abstraction isn't stricly necessary.
    It's helpful for having this in one place
    and also to only have to do the type hinting fix in one place.
    """
    try:
        import tqdm.auto
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_default_progress_bar", requirement="tqdm"
        ) from exc

    return partial(tqdm.auto.tqdm, **kwargs)  # type: ignore # can't get tqdm and mypy to play nice


def figure_out_progress_bars(
    progress_results: bool | ProgressLike | None,
    progress_results_default_kwargs: dict[str, Any],
    executor: Any | None,
    progress_parallel_submission: bool | ProgressLike | None,
    progress_parallel_submission_default_kwargs: dict[str, Any],
) -> tuple[ProgressLike | None, ProgressLike | None]:
    """
    Figure out which progress bars to use

    This is just a helper to avoid having to repeat this logic everywhere.

    Parameters
    ----------
    progress_results
        Progress bar to use for the results.

        If `True`, we use the default bar.

        Otherwise, we use the supplied value
        or `None` if the supplied value is falsey.

    progress_results_default_kwargs
        Keyword-arguments to pass to [`get_tqdm_auto`][(m).]
        if we are using the default bar (i.e. `progress_results` is `True`).

    executor
        Parallel executor being used.

        If this is `None`, we know that no parallelisation will occur
        so we can safely return `None` for `progress_parallel_submission`.

    progress_parallel_submission
        Progress bar to use for submitting the iterable to the parallel pool.

        If `True`, we use the default bar.

        Otherwise, we use the supplied value
        or `None` if the supplied value is falsey.

    progress_parallel_submission_default_kwargs
        Keyword-arguments to pass to [`get_tqdm_auto`][(m).]
        if we are using the default bar (i.e. `progress_parallel_submission` is `True`).

    Returns
    -------
    progress_results_use :
        The progress bar (or lack thereof) to use for result retrieval

    progress_parallel_submission_use :
        The progress bar (or lack thereof) to use for submitting to the parallel pool
    """
    if executor is not None:
        if progress_parallel_submission and isinstance(
            progress_parallel_submission, bool
        ):
            progress_parallel_submission_use = get_tqdm_auto(
                **progress_parallel_submission_default_kwargs
            )
        elif not progress_parallel_submission:
            progress_parallel_submission_use = None
        else:
            progress_parallel_submission_use = progress_parallel_submission
    else:
        progress_parallel_submission_use = None

    if progress_results and isinstance(progress_results, bool):
        progress_results_use = get_tqdm_auto(**progress_results_default_kwargs)
    elif not progress_results:
        progress_results_use = None
    else:
        progress_results_use = progress_results

    return progress_results_use, progress_parallel_submission_use


@define
class ParallelOpConfig:
    """
    Configuration for a potentially parallel op, potentially with a progress bar(s)
    """

    progress_results: ProgressLike | None = None
    """
    Progress bar to track the results of the op.

    If `None`, no progress bar is displayed for the results of the op.
    """

    # Note: I considered switching the executor for a Protocol here.
    # However, our parallelisation and progress bar display is tightly bound to
    # concurrent.futures' Future class.
    # I figure that, if a user wants to use another pattern,
    # they can and they will probably have other optimisations
    # they want to make too so I'm not going to try and make this too general now.
    executor: concurrent.futures.Executor | None = None
    """
    Executor with which to perform the op.

    If `None`, the op is executed serially.
    """

    progress_parallel_submission: ProgressLike | None = None
    """
    Progress bar to track the submission of the iterable to the parallel executor.

    If `None`, no progress bar is displayed for the results of the op.
    """

    executor_created_in_class_method: bool = False
    """
    Whether `self.executor` was created in a class method

    This can be used to indicate that the user needs to shutdown the executor,
    it was not created from an accessible place.
    """

    @classmethod
    def from_user_facing(
        cls,
        progress: bool = False,
        progress_results_kwargs: dict[str, Any] | None = None,
        progress_parallel_submission_kwargs: dict[str, Any] | None = None,
        max_workers: int | None = None,
        parallel_pool_cls: type[
            concurrent.futures.Executor
        ] = concurrent.futures.ProcessPoolExecutor,
    ):
        """
        Initialise from more user-facing arguments

        Parameters
        ----------
        progress
            Should we show progress bars?

        progress_results_kwargs
            Passed to [get_tqdm_auto][(m).] when creating the results bar.

            Only used if `progress`.

        progress_parallel_submission_kwargs
            Passed to [get_tqdm_auto][(m).] when creating the parallel submission bar.

            Only used if `progress` and `max_workers` is not `None`.

        max_workers
            Maximum number of parallel workers.

            If `None`, we set `executor` equal to `None` in the result.

        parallel_pool_cls
            Type of parallel pool executor to use if `max_workers` is not `None`.

        Returns
        -------
        :
            Initialised instance.
        """
        if progress:
            if progress_results_kwargs is None:
                progress_results_kwargs = {}

            progress_results = get_tqdm_auto(**progress_results_kwargs)

            if max_workers is not None:
                if progress_parallel_submission_kwargs is None:
                    progress_parallel_submission_kwargs = {}

                progress_parallel_submission = get_tqdm_auto(
                    **progress_parallel_submission_kwargs
                )

            else:
                progress_parallel_submission = None

        else:
            progress_results = progress_parallel_submission = None

        if max_workers is not None:
            executor = parallel_pool_cls(max_workers=max_workers)
            executor_created_in_class_method = True

        else:
            executor = None
            executor_created_in_class_method = False

        return cls(
            progress_results=progress_results,
            executor=executor,
            progress_parallel_submission=progress_parallel_submission,
            executor_created_in_class_method=executor_created_in_class_method,
        )


def apply_op_parallel_progress(
    func_to_call: Callable[Concatenate[U, P], T],
    iterable_input: Iterable[U],
    parallel_op_config: ParallelOpConfig,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[T, ...]:
    """
    Apply an operation, potentially in parallel and potentially with progress bars

    Parameters
    ----------
    func_to_call
        Operation to apply to (i.e. the function to call on) `iterable_input`.

    iterable_input
        The input on which to perform `func_to_call`.

        Each element of `iterable_input` is passed to `func_to_call`.

    parallel_op_config
        Configuration with which to execute the potentially parallel process

    *args
        Passed to each call to `func_to_call`.

    **kwargs
        Passed to each call to `func_to_call`.

    Returns
    -------
    :
        Results of each call to `func_to_call`.

    Notes
    -----
    This function does not handle nested progress bars in an obvious way.
    They are possible, but you'll need to set the bar creation stuff up
    appropriately in the layer above calling this function
    (this function doesn't need to know about the location of the bars,
    it just uses whatever bars it is given).

    This function also doesn't handle keeping track of order
    i.e. which outputs map to which inputs.
    If you need this, either add the injection suggested in the code below
    or create another function(
    e.g. `create apply_op_parallel_progress_output_tracker`).
    Our suggestion would be to do the latter first
    and only switch to injection if we find we need more flexibility,
    because the abstractions will likely become hard to follow.
    """
    if parallel_op_config.executor is None:
        # Run serially
        if parallel_op_config.progress_results:
            iterable_input = parallel_op_config.progress_results(iterable_input)

        # If you wanted to make output tracking injectable, something like
        # the below could allow that.
        #
        # res_l = [
        #   output_generator(func_to_call, v, *args, **kwargs) for v in iterable_input
        # ]
        #
        # output_generator would be something like
        # output_generator: Callable[
        #    [Callable[Concatenate[U, P], T], U, P.args, P.kwargs], V
        # ]
        # return type becomes tuple[V, ...]
        #
        # Honestly, I'm not sure it's a helpful abstraction
        # so would just duplicate first
        # and only abstract later if it was really obvious that it was needed.
        res = tuple(func_to_call(v, *args, **kwargs) for v in iterable_input)

        return res

    if parallel_op_config.progress_parallel_submission:
        iterable_input = parallel_op_config.progress_parallel_submission(iterable_input)

    futures = [
        parallel_op_config.executor.submit(
            func_to_call,
            v,
            *args,
            **kwargs,
        )
        for v in iterable_input
    ]

    iterator_results = concurrent.futures.as_completed(futures)
    if parallel_op_config.progress_results:
        iterator_results = parallel_op_config.progress_results(
            iterator_results, total=len(futures)
        )

    res = tuple(ft.result() for ft in iterator_results)

    return res
