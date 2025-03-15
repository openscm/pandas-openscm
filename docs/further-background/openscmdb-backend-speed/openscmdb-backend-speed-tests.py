"""
Tests of the speed of various back-ends for OpenSCMDB
"""

from __future__ import annotations

import concurrent.futures
import json
import platform
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import git
import numpy as np
import pandas as pd
import tqdm.auto

import pandas_openscm
from pandas_openscm.db import (
    DATA_BACKENDS,
    INDEX_BACKENDS,
    OpenSCMDB,
)
from pandas_openscm.parallelisation import ParallelOpConfig
from pandas_openscm.testing import create_test_df

HERE = Path(__file__).parent
GIT_REPO = git.Repo(search_parent_directories=True)

# Enable tqdm pandas extension
tqdm.auto.tqdm.pandas()


class Timer:
    """Timer helper"""

    def __enter__(self):
        """Start the timer within a context block"""
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        """Exit the context block and record the time"""
        self.time = perf_counter() - self.start


def run_test(  # noqa: PLR0913, PLR0915
    *,
    test_case: str,
    n_scenarios: int,
    backend: str,
    index_as_category_type: bool,
    max_workers: int,
    groupby: list[str],
    n_variables: int,
) -> tuple[dict[str, Any], Path]:
    """
    Run our speed tests
    """
    platform_info = platform.platform()

    units = ["K", "W", "yr"]

    if test_case == "magicc_full_output":
        big_df = create_test_df(
            n_scenarios=n_scenarios,
            n_runs=600,
            timepoints=np.arange(1750.0, 2150.0 + 1.0),
            variables=[
                (f"variable_{i}", units[i % len(units)]) for i in range(n_variables)
            ],
        )

    elif test_case == "magicc_future_quantiles":
        # More realistic use case
        big_df = create_test_df(
            n_scenarios=n_scenarios,
            n_runs=15,  # approximate n quantiles we carry around
            timepoints=np.arange(2025.0, 2150.0 + 1.0),
            variables=[
                (f"variable_{i}", units[i % len(units)]) for i in range(n_variables)
            ],
        )

    else:
        raise NotImplementedError(test_case)

    db_dir = Path(tempfile.mkdtemp())
    db = OpenSCMDB(
        db_dir=db_dir,
        backend_data=DATA_BACKENDS.get_instance(backend),
        backend_index=INDEX_BACKENDS.get_instance(backend),
    )

    scratch_dir = Path(tempfile.mkdtemp())

    test_save_index_not_db = scratch_dir / f"test-index-direct{db.backend_index.ext}"

    idx_out = pd.DataFrame(
        np.full(big_df.shape[0], 0), index=big_df.index, columns=["file_map"]
    )
    with Timer() as timer_save_index_not_db:
        # Not a 100% fair test as this is just serialisation time,
        # but gives a sense of overhead.
        db.backend_index.save_index(idx_out, test_save_index_not_db)

    # Every timeseries in its own file
    file_map_worst_case = pd.Series(
        [str(scratch_dir / f"filename{db.backend_data.ext}")] * big_df.shape[0],
        index=range(big_df.shape[0]),
    )

    test_save_file_map = scratch_dir / f"test-file-map{db.backend_index.ext}"

    with Timer() as timer_save_file_map_not_db:
        # Not a 100% fair test as this is just serialisation time,
        # but gives a sense of overhead.
        db.backend_index.save_file_map(file_map_worst_case, test_save_file_map)

    memory_in_mb = big_df.memory_usage(deep=True).sum() / 1024**2
    # I'm not convinced this works properly
    index_memory_in_kb = big_df.index.memory_usage(deep=True) / 1024
    index_as_frame_memory_in_kb = (
        big_df.index.to_frame(index=False).memory_usage(deep=True).sum() / 1024
    )

    test_all_in_one_file = scratch_dir / f"test-all-in-one{db.backend_data.ext}"
    with Timer() as timer_single_write:
        db.backend_data.save_data(big_df, test_all_in_one_file)

    test_single_group_file = scratch_dir / f"test-single-group{db.backend_data.ext}"
    with Timer() as timer_groupby_write:
        big_df.groupby(groupby, observed=True).progress_apply(
            db.backend_data.save_data, data_file=test_single_group_file
        )

    with Timer() as timer_db_groups_write:
        db.save(
            big_df,
            groupby=groupby,
            progress=True,
        )

    if backend == "in_memory":
        # Parallel not relevant
        timer_db_groups_write_parallel = Timer()
        timer_db_groups_write_parallel.time = None
    else:
        # Delete before writing in parallel
        db.delete()

        with Timer() as timer_db_groups_write_parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                db.save(
                    big_df,
                    groupby=groupby,
                    parallel_op_config_save=ParallelOpConfig(
                        executor=executor,
                        progress_results=tqdm.auto.tqdm,
                        progress_parallel_submission=tqdm.auto.tqdm,
                    ),
                )

    with Timer() as timer_db_load:
        db.load(progress=True)

    if backend == "in_memory":
        # Parallel not relevant
        timer_db_load_parallel = Timer()
        timer_db_load_parallel.time = None
    else:
        with Timer() as timer_db_load_parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                db.load(
                    parallel_op_config=ParallelOpConfig(
                        executor=executor,
                        progress_results=tqdm.auto.tqdm,
                        progress_parallel_submission=tqdm.auto.tqdm,
                    ),
                )

    with Timer() as timer_delete:
        db.delete()

    commit = GIT_REPO.head.object.hexsha
    if GIT_REPO.is_dirty():
        commit = f"{commit}-dirty"

    res = {
        "platform_info": platform_info,
        "python_version": platform.python_version(),
        "pandas-openscm_commit": commit,
        "pandas-openscm_version": pandas_openscm.__version__,
        "test_case": test_case,
        "index_as_category_type": index_as_category_type,
        "n_scenarios": n_scenarios,
        "backend": backend,
        "max_workers": max_workers,
        "groupby": groupby,
        "n_variables": n_variables,
        "n_timeseries": big_df.shape[0],
        "n_time_points": big_df.shape[1],
        "data_size_in_memory_MB": memory_in_mb,
        "index_size_in_memory_kb": index_memory_in_kb,
        "index_size_as_frame_in_memory_kb": index_as_frame_memory_in_kb,
        "time_save_index_not_db": timer_save_index_not_db.time,
        "time_save_file_map_not_db": timer_save_file_map_not_db.time,
        "time_all_in_one_write": timer_single_write.time,
        "time_groupby_write": timer_groupby_write.time,
        "time_db_grouped_write": timer_db_groups_write.time,
        "time_db_grouped_write_parallel": timer_db_groups_write_parallel.time,
        "time_db_load": timer_db_load.time,
        "time_db_load_parallel": timer_db_load_parallel.time,
        "time_delete": timer_delete.time,
    }
    out_json_stem = "_".join(
        [
            str(v)
            for v in [
                "openscmdb-backend-speed-test",
                backend,
                test_case,
                index_as_category_type,
                n_scenarios,
                max_workers,
                "-".join(groupby),
                n_variables,
            ]
        ]
    )

    out_json_name = f"{out_json_stem}.json"
    out_json_path = HERE / out_json_name
    return res, out_json_path


def run_tests() -> None:
    """Run the analysis"""
    # Fine to hard-code for now, we test enough other dimensions
    max_workers = 4
    groupby = ["scenario", "variable"]
    n_variables = 5

    test_res_l = []
    for test_case, n_scenarios in (
        ("magicc_full_output", 1),
        # ("magicc_full_output", 10),
        # ("magicc_full_output", 30),
        # ("magicc_full_output", 100),
        ("magicc_future_quantiles", 1),
        # ("magicc_future_quantiles", 100),
        # ("magicc_future_quantiles", 300),
        # ("magicc_future_quantiles", 1000),
    ):
        for backend in [v[0] for v in DATA_BACKENDS.options]:
            for index_as_category_type in [True, False]:
                msg = (
                    f"Running {test_case=} {n_scenarios=} {backend=}\n"
                    f"{index_as_category_type=}"
                )
                print(msg)

                test_res_l.append(
                    run_test(
                        test_case=test_case,
                        n_scenarios=n_scenarios,
                        backend=backend,
                        index_as_category_type=index_as_category_type,
                        max_workers=max_workers,
                        groupby=groupby,
                        n_variables=n_variables,
                    )
                )

    print("Dumping results to disk")
    for res, out_json_path in test_res_l:
        with open(out_json_path, "w") as fh:
            json.dump(res, fh, indent=2, sort_keys=True)
            fh.write("\n")

        print(f"Wrote {out_json_path}")


def main() -> None:
    """Run the main script"""
    run_tests()
    # write_summaries()


if __name__ == "__main__":
    main()
