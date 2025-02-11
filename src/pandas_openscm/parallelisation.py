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
[this excellent Stack Overflow answer](https://stackoverflow.com/a/60644649).
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, Callable, TypeAlias, TypeVar

from typing_extensions import Concatenate, ParamSpec

from pandas_openscm.exceptions import MissingOptionalDependencyError

P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")
V = TypeVar("V")

ProgressLike: TypeAlias = Callable[[Iterable[V]], Iterator[V]]
"""A callable that acts like something which creates a progress bar"""


def get_default_progress_bar(**kwargs: Any) -> ProgressLike[Any]:
    # Very basic
    # Abstraction is helpful for type hinting as well as just cleanliess
    try:
        import tqdm.auto
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_default_progress_bar", requirement="tqdm"
        ) from exc

    return partial(tqdm.auto.tqdm, **kwargs)  # type: ignore # can't get tqdm and mypy to play nice


def figure_out_progress_bars(
    progress_results: bool | ProgressLike[Any] | None,
    progress_results_default_kwargs: dict[str, Any],
    executor: Any | None,
    progress_parallel_submission: bool | ProgressLike[Any] | None,
    progress_parallel_submission_default_kwargs: dict[str, Any],
) -> tuple[ProgressLike[Any] | None, ProgressLike[Any] | None]:
    if executor is not None:
        if progress_parallel_submission and isinstance(
            progress_parallel_submission, bool
        ):
            progress_parallel_submission_use = get_default_progress_bar(
                **progress_parallel_submission_default_kwargs
            )
        elif not progress_parallel_submission:
            progress_parallel_submission_use = None
        else:
            progress_parallel_submission_use = progress_parallel_submission
    else:
        progress_parallel_submission_use = None

    if progress_results and isinstance(progress_results, bool):
        progress_results_use = get_default_progress_bar(
            **progress_results_default_kwargs
        )
    elif not progress_results:
        progress_results_use = None
    else:
        progress_results_use = progress_results

    return progress_results_use, progress_parallel_submission_use


def apply_op_parallel_progress(
    func_to_call: Callable[Concatenate[U, P], T],
    iterable_input: Iterable[U],
    progress_results: ProgressLike[Any] | None = None,
    progress_parallel_submission: ProgressLike[U] | None = None,
    # Note: I considered switching the executor for a Protocol here.
    # However, our parallelisation and progress bar display is tightly bound to
    # concurrent.futures' Future class.
    # I figure that, if a user wants to use another pattern,
    # they can and they will probably have other optimisations
    # they want to make too so I'm not going to try and make this too general now.
    executor: concurrent.futures.Executor | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[T, ...]:
    # Things this does do:
    # - apply op in parallel
    # - progress bar for overall progress
    #
    # Things this doesn't do:
    # - nested progress bars in an obvious way.
    #   However, they are possible,
    #   just include the bar creation stuff in the input iterable,
    #   i.e. do it in the layer above this function
    #   and then the function can just create the bar in the right position
    #   when it's called (this function doesn't need to know about
    #   nested progress bars).
    # - keep track of ordering/which outputs map to which inputs
    #   (if you need this, either add the injection suggested below
    #   or create apply_op_parallel_progress_output_tracker
    #   (the latter is our first suggestion, only do the injection
    #   if we find ourselves copying this lots as it could
    #   become very hard to follow).

    if executor is None:
        # Run serially
        if progress_results:
            iterable_input = progress_results(iterable_input)

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
        # Honestly, I'm not sure it's a helpful abstraction...
        res_g = (func_to_call(v, *args, **kwargs) for v in iterable_input)

        return tuple(res_g)

    if progress_parallel_submission:
        iterable_input = progress_parallel_submission(iterable_input)

    futures = [
        executor.submit(
            func_to_call,
            v,
            *args,
            **kwargs,
        )
        for v in iterable_input
    ]

    iterator_results = concurrent.futures.as_completed(futures)
    if progress_results:
        iterator_results = progress_results(iterator_results)

    res_g = (ft.result() for ft in iterator_results)

    return tuple(res_g)
