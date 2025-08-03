"""
Tests of the speed of various back-ends for OpenSCMDB
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import os
import platform
import tempfile
import textwrap
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import git
import numpy as np
import pandas as pd
import tqdm.auto
from attrs import define

import pandas_openscm
from pandas_openscm.db import DATA_BACKENDS, INDEX_BACKENDS, OpenSCMDB
from pandas_openscm.parallelisation import ParallelOpConfig
from pandas_openscm.testing import create_test_df

BACKEND_OPTIONS = [v[0] for v in DATA_BACKENDS.options]
GIT_REPO = git.Repo(search_parent_directories=True)
HERE = Path(__file__).parent
PLATFORM_INFO = platform.platform()

COMMIT = GIT_REPO.head.object.hexsha
if GIT_REPO.is_dirty():
    COMMIT = f"{COMMIT}-dirty"

pandas_openscm.register_pandas_accessors()


@define
class TestCase:
    """A speed test case"""

    backend: str
    """Back-end to use in the test"""

    id: str
    """ID for the test case"""

    variables: tuple[tuple[str, str], ...]
    """Variables to use in the test data"""

    n_scenarios: int
    """Number of scenarios to use in the test data"""

    n_runs: int
    """Number of runs to use in the test data"""

    time_points: np.typing.NDArray[np.floating]
    """Time points to use in the test data"""

    index_as_category_type: bool
    """Is df's index a category type"""

    max_workers: int
    """Maximum number of workers to use in parallel tests"""

    groupby: list[str] | None
    """Grouping to use when saving data"""

    @property
    def n_time_points(self) -> int:
        """Number of time points in self.df"""
        return self.time_points.size

    @property
    def n_time_series(self) -> int:
        """Number of time series in self.df"""
        return self.n_scenarios * self.n_runs * len(self.variables)

    @property
    def out_file(self) -> Path:
        """Path in which to write the results"""
        out_stem = "_".join(
            [
                str(v)
                for v in [
                    "openscmdb-backend-speed-test",
                    self.id,
                    self.backend,
                    self.n_time_series,
                    self.n_time_points,
                    self.index_as_category_type,
                    self.max_workers,
                    "-".join(self.groupby) if self.groupby is not None else "-",
                ]
            ]
        )

        out_name = f"{out_stem}.json"
        out_path = HERE / out_name

        return out_path

    def get_db_instance(self) -> OpenSCMDB:
        """Get a database instance"""
        db_dir = Path(tempfile.mkdtemp())
        db = OpenSCMDB(
            db_dir=db_dir,
            backend_data=DATA_BACKENDS.get_instance(self.backend),
            backend_index=INDEX_BACKENDS.get_instance(self.backend),
        )

        return db

    def get_df_instance(self) -> pd.DataFrame:
        """Get the test DataFrame"""
        res = create_test_df(
            variables=self.variables,
            n_scenarios=self.n_scenarios,
            n_runs=self.n_runs,
            timepoints=self.time_points,
        )
        if self.index_as_category_type:
            res = res.openscm.to_category_index()

        return res


class Timer:
    """Timer helper"""

    def __enter__(self):
        """Start the timer within a context block"""
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        """Exit the context block and record the time"""
        self.time = time.perf_counter() - self.start


def generate_test_cases(  # noqa: PLR0913
    high_level_combos: Iterable[tuple[list[str] | None, int, bool, str]],
    id: str,
    n_runs: int,
    n_variables: int,
    time_points: np.typing.NDArray[np.floating],
    max_workers: int,
    units: tuple[str, ...] = ("K", "W", "yr", "K / yr", "W yr"),
) -> tuple[TestCase, ...]:
    """Generate test cases"""
    res_l = []
    for groupby, n_scenarios, index_as_category_type, backend in high_level_combos:
        test_case = TestCase(
            backend=backend,
            id=id,
            variables=[
                (f"variable_{i}", units[i % len(units)]) for i in range(n_variables)
            ],
            n_scenarios=n_scenarios,
            n_runs=n_runs,
            time_points=time_points,
            index_as_category_type=index_as_category_type,
            max_workers=max_workers,
            groupby=groupby,
        )

        res_l.append(test_case)

    return tuple(res_l)


def run_test(test_case: TestCase, force: bool) -> Path:
    """Run a single test case"""
    if not force and test_case.out_file.exists():
        print(f"{test_case.out_file} already exists")
        return test_case.out_file

    print(f"Generating data for {test_case.out_file}")
    db = test_case.get_db_instance()
    df = test_case.get_df_instance()
    memory_in_mb = df.memory_usage(deep=True).sum() / 1024**2
    # I'm not convinced this works properly
    index_memory_in_kb = df.index.memory_usage(deep=True) / 1024
    index_as_frame_memory_in_kb = (
        df.index.to_frame(index=False).memory_usage(deep=True).sum() / 1024
    )

    print("Save")
    with Timer() as timer_save:
        db.save(
            df,
            groupby=test_case.groupby,
            progress=True,
        )

    db_size_on_disk = sum(
        os.path.getsize(f) for f in db.db_dir.glob("*") if f.is_file()
    )
    db_size_on_disk_mb = db_size_on_disk / 1024**2

    if test_case.backend == "in_memory":
        # Parallel not relevant
        timer_save_parallel = Timer()
        timer_save_parallel.time = None
    else:
        # Delete before saving again
        db.delete()
        print("Save parallel")
        with Timer() as timer_save_parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=test_case.max_workers
            ) as executor:
                db.save(
                    df,
                    groupby=test_case.groupby,
                    parallel_op_config_save=ParallelOpConfig(
                        executor=executor,
                        progress_results=tqdm.auto.tqdm,
                        progress_parallel_submission=tqdm.auto.tqdm,
                    ),
                )

    print("Load")
    with Timer() as timer_load:
        db.load(progress=True)

    if test_case.backend == "in_memory":
        # Parallel not relevant
        timer_load_parallel = Timer()
        timer_load_parallel.time = None
    else:
        print("Load parallel")
        with Timer() as timer_load_parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=test_case.max_workers
            ) as executor:
                db.load(
                    parallel_op_config=ParallelOpConfig(
                        executor=executor,
                        progress_results=tqdm.auto.tqdm,
                        progress_parallel_submission=tqdm.auto.tqdm,
                    ),
                )

    print("Delete")
    with Timer() as timer_delete:
        db.delete(progress=True)

    if test_case.groupby is None:
        n_db_files = 1
    else:
        n_db_files = len(df.groupby(test_case.groupby, observed=True))

    res = {
        "platform_info": PLATFORM_INFO,
        "python_version": platform.python_version(),
        "pandas-openscm_commit": COMMIT,
        "pandas-openscm_version": pandas_openscm.__version__,
        "id": test_case.id,
        "index_as_category_type": test_case.index_as_category_type,
        "n_scenarios": test_case.n_scenarios,
        "backend": test_case.backend,
        "max_workers": test_case.max_workers,
        "groupby": test_case.groupby,
        "n_variables": len(test_case.variables),
        "n_timeseries": df.shape[0],
        "n_time_points": df.shape[1],
        "n_db_files": n_db_files,
        "db_size_on_disk_MB": db_size_on_disk_mb,
        "data_size_in_memory_MB": memory_in_mb,
        "index_size_in_memory_kb": index_memory_in_kb,
        "index_size_as_frame_in_memory_kb": index_as_frame_memory_in_kb,
        "save_time": timer_save.time,
        "save_time_parallel": timer_save_parallel.time,
        "load_time": timer_load.time,
        "load_time_parallel": timer_load_parallel.time,
        "delete_time": timer_delete.time,
    }
    with open(test_case.out_file, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {test_case.out_file}")

    return test_case.out_file


def run_tests(test_cases: tuple[TestCase, ...], force: bool) -> tuple[Path, ...]:
    """Run tests"""
    res_l = []
    for test_case in test_cases:
        res_file = run_test(test_case, force=force)
        res_l.append(res_file)
        print()
        print()

    return tuple(res_l)


def get_col_expect_unique(indf: pd.DataFrame, col: str) -> Any:
    """Get a column value, expecting it to be unique"""
    col_vs = indf[col].unique().tolist()
    if len(col_vs) != 1:
        raise AssertionError(col_vs)

    return col_vs[0]


def write_high_level_info(db: pd.DataFrame, out_file: Path) -> None:
    """Write the high-level information"""
    pandas_openscm_version = get_col_expect_unique(db, "pandas-openscm_version")
    pandas_openscm_commit = get_col_expect_unique(db, "pandas-openscm_commit")
    python_version = get_col_expect_unique(db, "python_version")
    platform_info = get_col_expect_unique(db, "platform_info")
    n_workers = get_col_expect_unique(db, "max_workers")

    with open(out_file, "w") as fh:
        fh.write(
            "\n".join(
                [
                    f"- pandas-openscm version: {pandas_openscm_version}",
                    f"- pandas-openscm commit: {pandas_openscm_commit}",
                    f"- python version: {python_version}",
                    f"- platform information: {platform_info}",
                    f"- number of workers used in parallel tests: {n_workers}",
                ]
            )
        )
        fh.write("\n")


def write_performance_tables(
    res: pd.DataFrame, out_file: Path, thousands_sep: str = ","
) -> None:
    """Write tables documenting performance"""
    res = res.copy()

    def rename_backend(b: str) -> str:
        if b == "in_memory":
            return "In memory"

        return b

    res["backend"] = res["backend"].map(rename_backend)
    tabs_map = {
        "save_time": "Save time (s)",
        "save_time_parallel": "Save time parallel (s)",
        "db_size_on_disk_MB": "Size of database on disk (MB)",
        "load_time": "Load time (s)",
        "load_time_parallel": "Load time parallel (s)",
        "delete_time": "Delete time (s)",
    }

    backend_col_name = "Back-end"
    index_col_renamings = {
        "data_size_in_memory_MB": "In-memory size (MB)",
        # "index_size_as_frame_in_memory_kb": "In-memory index size (kB)",
        "n_timeseries": "Time series",
        "n_time_points": "Time points",
        "backend": backend_col_name,
        "n_db_files": "Files in database (after grouping)",
    }
    res = res.rename({**index_col_renamings}, axis="columns")
    res = res.set_index([*index_col_renamings.values()])

    indent = "    "
    with open(out_file, "w") as fh:
        for i, (operation, header) in enumerate(tabs_map.items()):
            fh.write(f'=== "{header}"\n\n')

            write_table = res[operation].unstack(backend_col_name).sort_index()
            html_table = (
                write_table.style.format(precision=2, thousands=thousands_sep)
                .format_index(precision=1, thousands=thousands_sep)
                .to_html()
            )

            fh.write(textwrap.indent(html_table, indent))

            if "In memory" in write_table and write_table["In memory"].isnull().any():
                msg = (
                    "*Note that there are no values for parallel operations "
                    "with an in-memory back-end "
                    "because such a setup doesn't make sense "
                    "(you just duplicate the Python process for no gain).\n"
                )
                fh.write("\n")
                fh.write(textwrap.indent(msg, indent))

            if i < len(tabs_map) - 1:
                fh.write("\n\n")

    print(f"Wrote {out_file}")


def write_summaries(res_files: tuple[Path, ...]) -> None:
    """Write summaries into our docs"""
    db_l = []
    for file in HERE.glob("*.json"):
        with open(file) as fh:
            db_l.append(json.load(fh))

    db = pd.DataFrame(db_l)

    write_high_level_info(db, HERE / "high-level-info.txt")

    write_performance_tables(
        db[(db["id"] == "full_scm_output") & ~db["index_as_category_type"]],
        HERE / "full-scm-output.txt",
    )
    # No need for this, MultiIndex is category by default
    # write_performance_tables(
    #     db[(db["id"] == "full_scm_output") & db["index_as_category_type"]],
    #     HERE / "full-scm-output-category-index.txt",
    # )

    write_performance_tables(
        db[(db["id"] == "scm_future_quantiles_output") & ~db["index_as_category_type"]],
        HERE / "scm-future-quantiles-output.txt",
    )
    # write_performance_tables(
    #     db[(db["id"] == "scm_future_quantiles_output")
    #     & db["index_as_category_type"]],
    #     HERE / "scm-future-quantiles-output-category-index.txt",
    # )


def main() -> None:
    """Run the main script"""
    max_workers = 8

    full_scm_output_kwargs = dict(
        n_runs=600,
        n_variables=5,
        time_points=np.arange(1750.0, 2150.0 + 1.0),
        max_workers=max_workers,
        id="full_scm_output",
    )
    test_cases_full_scm_output = generate_test_cases(
        high_level_combos=itertools.product(
            (None, ["scenario", "variable"], ["scenario", "variable", "run", "unit"]),
            (1,),
            (False,),
            BACKEND_OPTIONS,
        ),
        **full_scm_output_kwargs,
    )
    test_cases_full_scm_output_not_all_combos = generate_test_cases(
        high_level_combos=itertools.product(
            (None, ["scenario", "variable"]),  # don't do individual files, too much
            (
                10,
                30,
                100,
            ),
            (False,),
            BACKEND_OPTIONS,
        ),
        **full_scm_output_kwargs,
    )

    scm_future_quantiles_output_kwargs = dict(
        n_runs=15,  # approx no. of quantiles we normally carry around
        time_points=np.arange(2025.0, 2150.0 + 1.0),
        max_workers=max_workers,
        id="scm_future_quantiles_output",
    )
    test_cases_future_quantiles_output = generate_test_cases(
        high_level_combos=itertools.product(
            (None, ["variable"]),
            (1, 100, 300, 1000, 10000),
            (False,),
            BACKEND_OPTIONS,
        ),
        n_variables=5,
        **scm_future_quantiles_output_kwargs,
    )
    test_cases_future_quantiles_output_more_variables = generate_test_cases(
        high_level_combos=itertools.product(
            (None, ["variable"]),
            (1, 100, 300, 1000),
            (False,),
            BACKEND_OPTIONS,
        ),
        n_variables=50,
        **scm_future_quantiles_output_kwargs,
    )

    test_cases = tuple(
        [
            *test_cases_full_scm_output,
            *test_cases_full_scm_output_not_all_combos,
            *test_cases_future_quantiles_output,
            *test_cases_future_quantiles_output_more_variables,
        ]
    )
    force = False
    # # Fast running set up
    # test_cases = [*test_cases[:3], *test_cases[-3:]]
    # force = True
    results_files = run_tests(test_cases, force=force)
    write_summaries(results_files)


if __name__ == "__main__":
    main()
