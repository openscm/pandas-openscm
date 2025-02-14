"""
Tests of parallelisation with `pandas_openscm.OpenSCMDB`
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pandas_openscm.db import CSVBackend, EmptyDBError, OpenSCMDB
from pandas_openscm.parallelisation import ParallelOpConfig
from pandas_openscm.testing import create_test_df

tqdm_auto = pytest.importorskip("tqdm.auto")


@pytest.mark.slow
@pytest.mark.parametrize(
    "executor_ctx_manager, executor_ctx_manager_kwargs",
    (
        pytest.param(nullcontext, dict(enter_result=None), id="serial"),
        pytest.param(nullcontext, dict(enter_result=2), id="default-executor"),
        pytest.param(
            concurrent.futures.ThreadPoolExecutor,
            dict(max_workers=4),
            id="inject-threading",
        ),
        # If you use fork context here, pytest may hang.
        # This is a gotcha of parallelism.
        pytest.param(
            concurrent.futures.ProcessPoolExecutor,
            dict(max_workers=4, mp_context=multiprocessing.get_context("spawn")),
            id="inject-process",
        ),
    ),
)
@pytest.mark.parametrize(
    "progress_kwargs",
    (
        pytest.param({}, id="no-progress"),
        pytest.param(
            dict(progress=True),
            id="default-progress",
        ),
        pytest.param(
            dict(
                progress=True,
                progress_results_kwargs=dict(ncols=200),
                progress_parallel_submission_kwargs=dict(leave=False),
            ),
            id="default-progress-custom-init-kwargs",
        ),
        pytest.param(
            dict(
                progress_results=partial(tqdm_auto.tqdm, desc="Custom retrive"),
                progress_parallel_submission=partial(
                    tqdm_auto.tqdm, desc="Custom submit"
                ),
            ),
            id="custom-progress",
        ),
    ),
)
def test_save_load_delete_parallel(
    tmpdir, progress_kwargs, executor_ctx_manager, executor_ctx_manager_kwargs
):
    db = OpenSCMDB(db_dir=Path(tmpdir), backend=CSVBackend())

    # TODO: parallel save
    df = create_test_df(
        n_scenarios=15,
        variables=[
            ("variable_1", "kg"),
            ("variable_2", "Mt"),
            ("variable_3", "m"),
        ],
        n_runs=600,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )
    for _, svdf in df.groupby(["scenario", "variable"]):
        db.save(svdf)

    with executor_ctx_manager(**executor_ctx_manager_kwargs) as executor:
        from_user_facing_kwargs = {}
        parallel_progress_kwargs = {}
        if isinstance(executor, int):
            from_user_facing_kwargs["max_workers"] = executor

        for k in [
            "progress",
            "progress_results_kwargs",
            "progress_parallel_submission_kwargs",
        ]:
            if k in progress_kwargs:
                from_user_facing_kwargs[k] = progress_kwargs[k]

        if from_user_facing_kwargs:
            parallel_op_config = ParallelOpConfig.from_user_facing(
                **from_user_facing_kwargs
            )
        else:
            parallel_op_config = ParallelOpConfig()

        if isinstance(executor, concurrent.futures.Executor):
            parallel_op_config.executor = executor

        if "progress_results" in progress_kwargs:
            parallel_op_config.progress_results = progress_kwargs["progress_results"]

        if "progress_parallel_submission" in progress_kwargs:
            parallel_op_config.progress_parallel_submission = progress_kwargs[
                "progress_parallel_submission"
            ]

        loaded = db.load(out_columns_type=df.columns.dtype, **parallel_progress_kwargs)
        pd.testing.assert_frame_equal(loaded, df, check_like=True)

        db.delete(**parallel_progress_kwargs)

    with pytest.raises(EmptyDBError):
        db.load_metadata()

    with pytest.raises(EmptyDBError):
        db.load()
