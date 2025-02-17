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
import pytest

from pandas_openscm.db import CSVDataBackend, CSVIndexBackend, EmptyDBError, OpenSCMDB
from pandas_openscm.parallelisation import ParallelOpConfig
from pandas_openscm.testing import assert_frame_alike, create_test_df

tqdm_auto = pytest.importorskip("tqdm.auto")


@pytest.mark.slow
@pytest.mark.parametrize(
    "max_workers",
    (
        pytest.param(None, id="serial"),
        pytest.param(1, id="one-worker"),
        pytest.param(4, id="four-workers"),
    ),
)
@pytest.mark.parametrize(
    "progress",
    (pytest.param(False, id="no-progress"), pytest.param(True, id="progress")),
)
def test_save_load_delete_parallel(tmpdir, progress, max_workers):
    # TODO: switch to InMemoryDataBackend
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=CSVDataBackend(),
        backend_index=CSVIndexBackend(),
    )

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

    db.save(
        df, groupby=["scenario", "variable"], progress=progress, max_workers=max_workers
    )

    loaded = db.load(
        out_columns_type=df.columns.dtype, progress=progress, max_workers=max_workers
    )
    assert_frame_alike(loaded, df)

    db.delete(progress=progress, max_workers=max_workers)

    with pytest.raises(EmptyDBError):
        db.load_metadata()

    with pytest.raises(EmptyDBError):
        db.load()


@pytest.mark.slow
@pytest.mark.parametrize(
    "executor_ctx_manager, executor_ctx_manager_kwargs",
    (
        pytest.param(nullcontext, dict(enter_result=None), id="serial"),
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
    "save_progress_kwargs, load_progress",
    (
        pytest.param(
            dict(
                progress_grouping=partial(tqdm_auto.tqdm, desc="Custom grouping"),
                parallel_op_config_save=ParallelOpConfig(
                    progress_results=partial(tqdm_auto.tqdm, desc="Custom save"),
                    progress_parallel_submission=partial(
                        tqdm_auto.tqdm, desc="Custom save submit"
                    ),
                ),
                parallel_op_config_delete=ParallelOpConfig(
                    progress_results=partial(tqdm_auto.tqdm, desc="Custom save"),
                    progress_parallel_submission=partial(
                        tqdm_auto.tqdm, desc="Custom save submit"
                    ),
                ),
                parallel_op_config_rewrite=ParallelOpConfig(
                    progress_results=partial(tqdm_auto.tqdm, desc="Custom save"),
                    progress_parallel_submission=partial(
                        tqdm_auto.tqdm, desc="Custom save submit"
                    ),
                ),
            ),
            ParallelOpConfig(
                progress_results=partial(tqdm_auto.tqdm, desc="Custom load"),
                progress_parallel_submission=partial(
                    tqdm_auto.tqdm, desc="Custom load submit"
                ),
            ),
            id="custom-progress",
        ),
    ),
)
def test_save_load_delete_parallel_custom_progress(
    tmpdir,
    save_progress_kwargs,
    load_progress,
    executor_ctx_manager,
    executor_ctx_manager_kwargs,
):
    # TODO: switch to InMemoryDataBackend
    db = OpenSCMDB(
        db_dir=Path(tmpdir),
        backend_data=CSVDataBackend(),
        backend_index=CSVIndexBackend(),
    )

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

    with executor_ctx_manager(**executor_ctx_manager_kwargs) as executor:
        if isinstance(executor, concurrent.futures.Executor):
            for k, v in save_progress_kwargs.items():
                if k.startswith("parallel_op_config"):
                    v.executor = executor

        elif executor is None:
            pass

        else:
            raise NotImplementedError(executor)

        db.save(df, groupby=["scenario", "variable"], **save_progress_kwargs)

        loaded = db.load(
            out_columns_type=df.columns.dtype, parallel_op_config=load_progress
        )
        assert_frame_alike(loaded, df)

        db.delete(parallel_op_config=save_progress_kwargs["parallel_op_config_delete"])

    with pytest.raises(EmptyDBError):
        db.load_metadata()

    with pytest.raises(EmptyDBError):
        db.load()
