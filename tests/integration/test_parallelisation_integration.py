"""
Tests of `pandas_openscm.parallelisation`
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing

import pytest

from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress
from pandas_openscm.testing import changer

tqdm = pytest.importorskip("tqdm")
tqdm_asyncio = pytest.importorskip("tqdm.asyncio")


def test_from_user_facing_simplest():
    """
    Test if you use `from_user_facing` with the simplest inputs
    """
    res = ParallelOpConfig.from_user_facing(progress=True, max_workers=3)

    results_bar = res.progress_results(["an", "iterable"])
    assert isinstance(results_bar, tqdm_asyncio.tqdm_asyncio)
    assert results_bar.desc == ""

    assert isinstance(res.executor, concurrent.futures.ProcessPoolExecutor)
    assert res.executor._max_workers == 3

    submission_bar = res.progress_parallel_submission(["an", "iterable"])
    assert isinstance(submission_bar, tqdm_asyncio.tqdm_asyncio)
    assert submission_bar.desc == ""

    assert res.executor_created_in_class_method

    res.executor.shutdown()


def test_from_user_facing_progress_bar_kwargs():
    """
    Test if you use `from_user_facing`, controlling some progress bar features
    """
    res = ParallelOpConfig.from_user_facing(
        progress=True,
        progress_results_kwargs=dict(desc="Results"),
        progress_parallel_submission_kwargs=dict(desc="Submission"),
        max_workers=4,
    )

    results_bar = res.progress_results(["an", "iterable"])
    assert isinstance(results_bar, tqdm_asyncio.tqdm_asyncio)
    assert results_bar.desc == "Results"

    assert isinstance(res.executor, concurrent.futures.ProcessPoolExecutor)
    assert res.executor._max_workers == 4

    submission_bar = res.progress_parallel_submission(["an", "iterable"])
    assert isinstance(submission_bar, tqdm_asyncio.tqdm_asyncio)
    assert submission_bar.desc == "Submission"

    assert res.executor_created_in_class_method

    res.executor.shutdown()


def test_from_user_facing_progress_only():
    """
    Test if you use `from_user_facing`, just turning on progress, nothing else
    """
    res = ParallelOpConfig.from_user_facing(progress=True)

    results_bar = res.progress_results(["an", "iterable"])
    assert isinstance(results_bar, tqdm_asyncio.tqdm_asyncio)
    assert results_bar.desc == ""

    assert res.executor is None

    assert res.progress_parallel_submission is None

    assert not res.executor_created_in_class_method


def test_from_user_facing_parallel_only():
    """
    Test if you use `from_user_facing`, just turning on progress, nothing else
    """
    res = ParallelOpConfig.from_user_facing(max_workers=4)

    assert res.progress_results is None

    assert isinstance(res.executor, concurrent.futures.ProcessPoolExecutor)
    assert res.executor._max_workers == 4

    assert res.progress_parallel_submission is None

    assert res.executor_created_in_class_method

    res.executor.shutdown()


apply_op_parallel_progress_args_kwargs = pytest.mark.parametrize(
    "args, kwargs",
    (
        pytest.param(
            [2.0],
            {},
            id="no-kwargs",
        ),
        pytest.param(
            [2.0],
            {"exponent": 1.2},
            id="kwargs",
        ),
    ),
)


@pytest.mark.parametrize(
    "parallel_op_config",
    (
        pytest.param(
            ParallelOpConfig(
                progress_results=None,
                executor=None,
                progress_parallel_submission=None,
            ),
            id="basic",
        ),
        pytest.param(
            ParallelOpConfig.from_user_facing(progress=True),
            id="progress-only",
        ),
        pytest.param(
            ParallelOpConfig(
                progress_results=tqdm.tqdm,
                executor=None,
                progress_parallel_submission=None,
            ),
            id="progress-results-only",
        ),
    ),
)
@apply_op_parallel_progress_args_kwargs
def test_apply_op_parallel_progress_no_executor(parallel_op_config, args, kwargs):
    res = apply_op_parallel_progress(
        changer,
        range(4),
        parallel_op_config,
        *args,
        **kwargs,
    )

    assert isinstance(res, tuple)

    exp = tuple(changer(v, *args, **kwargs) for v in range(4))

    assert res == exp


@pytest.mark.parametrize(
    "executor_cls, executor_cls_kwargs",
    (
        pytest.param(
            concurrent.futures.ProcessPoolExecutor,
            # Have to use spawn within pytest
            dict(mp_context=multiprocessing.get_context("spawn")),
            id="process-pool-spwan",
        ),
        pytest.param(
            concurrent.futures.ThreadPoolExecutor,
            {},
            id="thread-pool",
        ),
    ),
)
@pytest.mark.parametrize(
    "progress_results",
    (
        pytest.param(None, id="no-progress-results"),
        pytest.param(tqdm.tqdm, id="progress-results"),
    ),
)
@pytest.mark.parametrize(
    "progress_parallel_submission",
    (
        pytest.param(None, id="no-progress-parallel-submission"),
        pytest.param(tqdm.tqdm, id="progress-parallel-submission"),
    ),
)
@apply_op_parallel_progress_args_kwargs
def test_apply_op_parallel_progress_our_executor(  # noqa: PLR0913
    executor_cls,
    executor_cls_kwargs,
    progress_results,
    progress_parallel_submission,
    args,
    kwargs,
):
    with executor_cls(**executor_cls_kwargs) as executor:
        parallel_op_config = ParallelOpConfig(
            progress_results=progress_results,
            executor=executor,
            progress_parallel_submission=progress_parallel_submission,
        )

        res = apply_op_parallel_progress(
            changer,
            range(4),
            parallel_op_config,
            *args,
            **kwargs,
        )

    assert isinstance(res, tuple)

    exp = tuple(changer(v, *args, **kwargs) for v in range(4))

    # Order not guaranteed with parallel runs
    assert set(res) == set(exp)
