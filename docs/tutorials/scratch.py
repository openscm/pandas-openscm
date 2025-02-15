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
import os.path
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm.autonotebook as tqdman

from pandas_openscm.db import FeatherBackend, OpenSCMDB
from pandas_openscm.parallelisation import ParallelOpConfig
from pandas_openscm.testing import create_test_df

# %%
units = ["K", "W", "yr"]
big_df = create_test_df(
    n_scenarios=250,
    n_runs=60,
    timepoints=np.arange(1750.0, 2100.0 + 1.0),
    variables=[(f"variable_{i}", units[i % len(units)]) for i in range(4)],
)
big_df

# %%
# How big is this thing
big_df.info(memory_usage="deep")

# %%
groupby = ["scenario", "variable"]

# %%
db_dir = Path(tempfile.mkdtemp())
# db = OpenSCMDB(db_dir=db_dir, backend=CSVBackend())
db = OpenSCMDB(db_dir=db_dir, backend=FeatherBackend())
# db = OpenSCMDB(db_dir=db_dir, backend=netCDFBackend())
db

# %%
scratch_dir = Path(tempfile.mkdtemp())

# %%
# %%time
# How long does it take to just write as a single file
test_all_in_one_file = scratch_dir / f"test-all-in-one{db.backend.ext}"
db.backend.save_data(big_df.reset_index(), test_all_in_one_file)
os.path.getsize(test_all_in_one_file) / (2**10) ** 2

# %%
# %%time
# Grouping makes things an incredible amount slower.
# Really slow for writing, but way faster for later reading.
test_single_group_file = scratch_dir / f"test-single-group{db.backend.ext}"
for _, df in tqdman.tqdm(big_df.groupby(groupby)):
    db.backend.save_data(df, test_single_group_file)

# %%
max_workers = 4

# %%
# Make a database to work with
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    db.save(
        big_df,
        groupby=groupby,
        progress_grouping=partial(tqdman.tqdm, desc="Grouping"),
        parallel_op_config_save=ParallelOpConfig(
            progress_results=partial(tqdman.tqdm, desc="Saving files"),
            progress_parallel_submission=partial(
                tqdman.tqdm, desc="Submitting files to save executor"
            ),
            executor=executor,
        ),
    )

# %%
# %%time
# How long does it take to load the index directly with pandas
pd.read_feather(db.index_file).shape

# %%
# %%time
# How long does it take to load the index via the db
index = db.load_index()
index

# %%
# %%time
# How long does it take to load the file map
file_map = db.load_file_map()
file_map

# %%
# %%time
# How long does it take to save the index directly with pandas
scratch_index_file_pd = scratch_dir / f"pd-index{db.backend.ext}"
index.to_feather(scratch_index_file_pd)

# %%
# %%time
# How long does it take to save the index via the db
scratch_index_file = scratch_dir / f"index{db.backend.ext}"
db.backend.save_index(index, scratch_index_file)

# %%
# %time
# How long does it take to just write the index as a single file
test_all_in_one_file_index = (
    Path(tempfile.mkdtemp()) / f"test-all-in-one-index{db.backend.ext}"
)
big_df.index.to_frame(index=False).to_feather(test_all_in_one_file_index)
os.path.getsize(test_all_in_one_file_index) / (2**10) ** 2

# %%
# %time
# How long does it take to just write the file map as a single file
test_all_in_one_file_map = (
    Path(tempfile.mkdtemp()) / f"test-all-in-one-map{db.backend.ext}"
)
# big_df.index.to_frame(index=False).to_feather(test_all_in_one_file_map)
# os.path.getsize(test_all_in_one_file_index) / (2 ** 10) ** 2

# %%
test_all_in_one_file_map

# %%
dump_dir = Path(tempfile.mkdtemp())
test_all_in_one_file = dump_dir / f"test-all-in-one{db.backend.ext}"
db.backend.save_data(big_df, test_all_in_one_file)
os.path.getsize(test_all_in_one_file) / (2**10) ** 2

# %%

# %%
tqdman.tqdm.pandas()

# %%
big_df.groupby(groupby).progress_apply(
    db.backend.save_data, data_file=test_single_group_file
)

# %%
max_workers = 4

# %%
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    db.save(
        big_df,
        groupby=groupby,
        progress_grouping=partial(tqdman.tqdm, desc="Grouping"),
        parallel_op_config_save=ParallelOpConfig(
            progress_results=partial(tqdman.tqdm, desc="Saving files"),
            progress_parallel_submission=partial(
                tqdman.tqdm, desc="Submitting files to save executor"
            ),
            executor=executor,
        ),
    )
    print(
        db.load(
            parallel_op_config=ParallelOpConfig(
                progress_results=partial(tqdman.tqdm, desc="Loading files"),
                progress_parallel_submission=partial(
                    tqdman.tqdm, desc="Submitting files to load executor"
                ),
                executor=executor,
            )
        ).shape
    )
    db.delete()

# %%
db.delete()

# %%
db

# %%
db.save(big_df, groupby=groupby, progress=True)
print(db.load(progress=True).shape)
db.delete(progress=True)

# %%
assert False, "stop"

# %%
db.save(big_df, groupby=groupby, progress=True)
print(db.load(progress=True).shape)
db.delete(progress=True)

# %%
max_workers = 4

# %%
db.save(big_df, groupby=groupby, progress=True, max_workers=max_workers)
print(db.load(progress=True, max_workers=max_workers).shape)
db.delete(progress=True, max_workers=max_workers)

# %%
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    para_op_conf = ParallelOpConfig(
        progress_results=partial(tqdman.tqdm, desc="Results"),
        progress_parallel_submission=partial(tqdman.tqdm, desc="Submission"),
        executor=executor,
    )

    db.save(
        big_df,
        groupby=groupby,
        progress_grouping=partial(tqdman.tqdm, desc="Grouping"),
        parallel_op_config_save=para_op_conf,
    )
    print(db.load(parallel_op_config=para_op_conf).shape)
    db.delete(parallel_op_config=para_op_conf)

# %%
# # Appears to kill Python when writing in parallel with netCDF, which is fun
# with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#     para_op_conf = ParallelOpConfig(
#         progress_results=partial(tqdman.tqdm, desc="Results"),
#         progress_parallel_submission=partial(tqdman.tqdm, desc="Submission"),
#         executor=executor,
#     )

#     db.save(big_df, groupby=groupby, progress_grouping=partial(tqdman.tqdm, desc="Grouping"), parallel_op_config_save=para_op_conf)
#     print(db.load(parallel_op_config=para_op_conf).shape)
#     db.delete(parallel_op_config=para_op_conf)

# %%
