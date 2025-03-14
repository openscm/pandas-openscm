# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import concurrent.futures
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm.auto

from pandas_openscm.db import (
    DATA_BACKENDS,
    INDEX_BACKENDS,
    OpenSCMDB,
)
from pandas_openscm.parallelisation import ParallelOpConfig
from pandas_openscm.testing import create_test_df

# %%
# Enable tqdm pandas extension
tqdm.auto.tqdm.pandas()

# %%
test_case = "magicc_full_output"
n_scenarios = 10
n_scenarios = 30
# test_case = "magicc_future_quantiles"
# n_scenarios = 100
backend = "csv"
# backend = "feather"
backend = "netCDF"
max_workers = 4
index_as_category_type = True

# %%
# Hard-code for now
groupby = ["scenario", "variable"]
n_variables = 5

# %%
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

# %%
if index_as_category_type:
    big_df = pd.DataFrame(
        big_df.values,
        index=pd.MultiIndex.from_frame(big_df.index.to_frame().astype("category")),
        columns=big_df.columns,
    )

# %%
db_dir = Path(tempfile.mkdtemp())
db = OpenSCMDB(
    db_dir=db_dir,
    backend_data=DATA_BACKENDS.get_instance(backend),
    backend_index=INDEX_BACKENDS.get_instance(backend),
)
db

# %%
scratch_dir = Path(tempfile.mkdtemp())

# %%
from time import perf_counter


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start


# %%
test_save_index_not_db = scratch_dir / f"test-index-direct{db.backend_index.ext}"

idx_out = pd.DataFrame(
    np.full(big_df.shape[0], 0), index=big_df.index, columns=["file_map"]
)
with Timer() as timer_save_index_not_db:
    # Not a 100% fair test as this is just serialisation time,
    # but gives a sense of overhead.
    db.backend_index.save_index(idx_out, test_save_index_not_db)

timer_save_index_not_db.time

# %%
test_save_index_not_db_category = (
    scratch_dir / f"test-index-direct-category{db.backend_index.ext}"
)

idx_out = pd.DataFrame(
    np.full(big_df.shape[0], 0),
    index=pd.MultiIndex.from_frame(big_df.index.to_frame().astype("category")),
    columns=["file_map"],
)
with Timer() as timer_save_index_not_db_category:
    # Not a 100% fair test as this is just serialisation time,
    # but gives a sense of overhead.
    db.backend_index.save_index(idx_out, test_save_index_not_db_category)

timer_save_index_not_db_category.time

# %%
# Every timeseries in its own file
file_map_worst_case = pd.Series(
    [str(scratch_dir / f"filename{db.backend_data.ext}")] * big_df.shape[0],
    index=range(big_df.shape[0]),
)
file_map_worst_case

# %%
test_save_file_map = scratch_dir / f"test-file-map{db.backend_index.ext}"

with Timer() as timer_save_file_map_not_db:
    # Not a 100% fair test as this is just serialisation time,
    # but gives a sense of overhead.
    db.backend_index.save_file_map(file_map_worst_case, test_save_file_map)

timer_save_file_map_not_db.time

# %%
memory_in_mb = big_df.memory_usage(deep=True).sum() / 1024**2
memory_in_mb

# %%
# I'm not convinced this works properly
index_memory_in_kb = big_df.index.memory_usage(deep=True) / 1024
index_memory_in_kb

# %%
index_as_frame_memory_in_kb = (
    big_df.index.to_frame(index=False).memory_usage(deep=True).sum() / 1024
)
index_as_frame_memory_in_kb

# %%
test_all_in_one_file = scratch_dir / f"test-all-in-one{db.backend_data.ext}"
with Timer() as timer_single_write:
    db.backend_data.save_data(big_df, test_all_in_one_file)

timer_single_write.time

# %%
test_single_group_file = scratch_dir / f"test-single-group{db.backend_data.ext}"
with Timer() as timer_groupby_write:
    big_df.groupby(groupby, observed=True).progress_apply(
        db.backend_data.save_data, data_file=test_single_group_file
    )

timer_groupby_write.time

# %%
with Timer() as timer_db_groups_write:
    db.save(
        big_df,
        groupby=groupby,
        progress=True,
    )

timer_db_groups_write.time

# %%
db.delete()

# %%
with Timer() as timer_db_groups_write_parallel:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        db.save(
            big_df,
            groupby=groupby,
            parallel_op_config_save=ParallelOpConfig(
                executor=executor,
                progress_results=tqdm.auto.tqdm,
                progress_parallel_submission=tqdm.auto.tqdm,
            ),
        )

timer_db_groups_write_parallel.time

# %%
with Timer() as timer_delete:
    db.delete()

timer_delete.time

# %%
res = {
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
    "time_save_index_not_db_as_category": timer_save_index_not_db_category.time,
    "time_save_file_map_not_db": timer_save_file_map_not_db.time,
    "time_all_in_one_write": timer_single_write.time,
    "time_groupby_write": timer_groupby_write.time,
    "time_db_grouped_write": timer_db_groups_write.time,
    "time_db_grouped_write_parallel": timer_db_groups_write_parallel.time,
    "time_delete": timer_delete.time,
}
out_json_stem = "_".join(
    [
        str(v)
        for v in [
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

res

# %%
out_json_name = f"{out_json_stem}.json"
with open(out_json_name, "w") as fh:
    json.dump(res, fh, indent=2, sort_keys=True)

out_json_name
