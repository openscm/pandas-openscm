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
import contextlib
import os.path
import sqlite3
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm.autonotebook as tqdman
import xarray as xr

from pandas_openscm.db import (
    CSVBackend,
    FeatherBackend,
    OpenSCMDB,
    netCDFBackend,
)
from pandas_openscm.parallelisation import ParallelOpConfig
from pandas_openscm.testing import create_test_df

# %%
db_dir = Path(tempfile.mkdtemp())
db = OpenSCMDB(db_dir=db_dir, backend=CSVBackend())
db = OpenSCMDB(db_dir=db_dir, backend=FeatherBackend())
db = OpenSCMDB(db_dir=db_dir, backend=netCDFBackend())
db

# %%
units = ["K", "W", "yr"]

test_case = "artificial"
test_case = "more_realistic"
# test_case = "like_for_like"

if test_case == "artificial":
    # Artificial as you'd almost never do analysis
    # on multiple scenarios with full ensembles all at once.
    # You'd always pre-process in some dimension first,
    # which is an embarassingly parallel problem.
    artificial_test_create_kwargs = dict(
        n_scenarios=10 if isinstance(db.backend, CSVBackend) else 200,  # CSV way slower
        n_runs=600,
        timepoints=np.arange(1750.0, 2100.0 + 1.0),
        variables=[(f"variable_{i}", units[i % len(units)]) for i in range(4)],
    )
    groupby = ["scenario", "variable"]
    big_df = create_test_df(**artificial_test_create_kwargs)

elif test_case == "more_realistic":
    # More realistic use case
    more_realistic_test_create_kwargs = dict(
        n_scenarios=75
        if isinstance(db.backend, CSVBackend)
        else 2_000,  # CSV way slower
        n_runs=15,  # number of quantiles we generally carry around
        timepoints=np.arange(1750.0, 2100.0 + 1.0),
        variables=[(f"variable_{i}", units[i % len(units)]) for i in range(30)],
    )
    groupby = ["variable"]
    big_df = create_test_df(**more_realistic_test_create_kwargs)

elif test_case == "like_for_like":
    # like-for-like to compare CSV and others directly
    like_for_like_test_create_kwargs = dict(
        n_scenarios=75,
        n_runs=15,
        timepoints=np.arange(1750.0, 2100.0 + 1.0),
        variables=[(f"variable_{i}", units[i % len(units)]) for i in range(30)],
    )
    groupby = ["variable"]
    big_df = create_test_df(**like_for_like_test_create_kwargs)

else:
    raise NotImplementedError(test_case)

# %%
groupby

# %%
big_df

# %%
# How big is this thing
big_df.info(memory_usage="deep")

# %%
scratch_dir = Path(tempfile.mkdtemp())

# %%
db.backend

# %%
# %%time
# How long does it take to just write as a single file
test_all_in_one_file = scratch_dir / f"test-all-in-one{db.backend.ext}"
db.backend.save_data(big_df, test_all_in_one_file)
os.path.getsize(test_all_in_one_file) / (2**10) ** 2

# %%
# %%time
if big_df.shape[0] <= 50_000:
    # How long does it take to just dump to sqlite
    test_sqlite = scratch_dir / "test-sqlite.db"
    test_sqlite_table_name = "big_df"

    with contextlib.closing(sqlite3.connect(test_sqlite)) as con:
        big_df.to_sql(name=test_sqlite_table_name, con=con, if_exists="replace")

    os.path.getsize(test_sqlite) / (2**10) ** 2
else:
    print("This will take ages with SQL, hence skipping")

# %%
# %%time
if big_df.shape[0] <= 50_000:
    with contextlib.closing(sqlite3.connect(test_sqlite)) as con:
        # Can't inject table name, that's fun
        loaded = pd.read_sql("SELECT * FROM big_df", con=con)

    print(loaded.info(memory_usage="deep"))

else:
    print("This will take ages with SQL, hence skipping")

# %%
# %%time
# How long does it take if we do a naive groupby.
test_single_group_file = scratch_dir / f"test-single-group{db.backend.ext}"
tqdman.tqdm.pandas()
big_df.groupby(groupby).progress_apply(
    db.backend.save_data, data_file=test_single_group_file
)

# %%
max_workers = 4

# %%
# %%time
# Make a database to work with
db.delete()
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
# Make a database to work with
if isinstance(db.backend, netCDFBackend):
    print("Threaded saving causes netCDF to break")
else:
    db.delete()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
# How long does it take to load the index directly
if isinstance(db.backend, CSVBackend):
    pd.read_csv(db.index_file)
elif isinstance(db.backend, netCDFBackend):
    xr.load_dataset(db.index_file)
else:
    pd.read_feather(db.index_file)

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
if isinstance(db.backend, CSVBackend):
    index.to_csv(scratch_index_file_pd)
elif isinstance(db.backend, netCDFBackend):
    print("not really a fair way to do this comparison because of sparsity")
else:
    index.to_feather(scratch_index_file_pd)

# %%
# %%time
# How long does it take to save the index directly with pandas as a category
scratch_index_file_pd = scratch_dir / f"pd-category-index{db.backend.ext}"
if isinstance(db.backend, CSVBackend):
    index.astype("category").to_csv(scratch_index_file_pd)
elif isinstance(db.backend, netCDFBackend):
    print("not really a fair way to do this comparison because of sparsity")
else:
    index.astype("category").to_feather(scratch_index_file_pd)

# %%
# %%time
# How long does it take to save the index directly with pandas to SQL
scratch_index_file_sql_pd = scratch_dir / "pd-sql.db"
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_pd)) as con:
    index.to_sql(
        name="index_table",
        con=con,
        if_exists="replace",
        index=True,
        index_label=index.index.names,
    )

# %%
index.info(memory_usage="deep")

# %%
# %%time
# How long does it take to read the index with SQL
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_pd)) as con:
    # Can't inject table name, that's fun
    sql_load = pd.read_sql(
        "SELECT * FROM index_table", con=con, index_col=index.index.names
    )
    print(f"{isinstance(sql_load.index, pd.MultiIndex)=}")
sql_load.info(memory_usage="deep")
sql_load

# %%
# %%time
# How long does it take to save the file map directly with pandas to SQL
scratch_index_file_sql_pd = scratch_dir / "pd-sql.db"
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_pd)) as con:
    file_map.to_sql(name="file_map", con=con, if_exists="replace")

# %%
file_map.info(memory_usage="deep")

# %%
# %%time
# How long does it take to read the index with SQL
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_pd)) as con:
    # Can't inject table name, that's fun
    sql_load = pd.read_sql("SELECT * FROM file_map", con=con)
    print(f"{isinstance(sql_load.index, pd.MultiIndex)=}")
sql_load.info(memory_usage="deep")

# %%
# %%time
# How long does it take to save the index via the db
scratch_index_file = scratch_dir / f"index{db.backend.ext}"
db.backend.save_index(index, scratch_index_file)

# %%
# %%time
# How long does it take to save the file map directly with pandas
scratch_file_map_file_pd = scratch_dir / f"pd-file-map{db.backend.ext}"
if isinstance(db.backend, CSVBackend):
    file_map.to_csv(scratch_file_map_file_pd)
elif isinstance(db.backend, netCDFBackend):
    file_map.to_xarray().to_netcdf(scratch_file_map_file_pd)
else:
    file_map.astype(str).to_frame().to_feather(scratch_file_map_file_pd)

# %%
# %%time
# How long does it take to save the file map via the db
scratch_file_map_file = scratch_dir / f"file-map{db.backend.ext}"
db.backend.save_file_map(file_map, scratch_file_map_file)

# %% [markdown]
# A more sensible SQL comparison

# %%
# %%time
# Set up the tables sensibly
scratch_index_file_sql_zn = scratch_dir / "pd-sql-zn.db"
if scratch_index_file_sql_zn.exists():
    os.remove(scratch_index_file_sql_zn)

with contextlib.closing(sqlite3.connect(scratch_index_file_sql_zn)) as con:
    cursor = con.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS db_index (
            run INTEGER,
            scenario TEXT,
            unit TEXT,
            variable TEXT,
            file_id INTEGER
        );
        """
    )
    for col in index.index.names:
        cursor.execute(
            # Hmmm not sure how this meant to be done without injection,
            # seems impossible to pass parameters.
            # Answer is probably that you're not meant to do this.
            # Pandas just do the formatting like this,
            # so I guess if you're certain about the source/control the source,
            # it's the only option:
            # https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/io/sql.py
            f"""
            CREATE INDEX idx_{col} ON db_index({col});
            """,
        )

    cursor.execute(
        """
        CREATE INDEX idx_composite ON db_index(run, scenario, unit, variable);
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS file_map (
            file_id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL
        );
        """
    )

# %%
# %%time
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_zn)) as con:
    file_map.to_sql(name="file_map", con=con, if_exists="append")

# %%
# %%time
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_zn)) as con:
    index.to_sql(name="db_index", con=con, if_exists="append")

# %%
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_zn)) as con:
    cursor = con.cursor()
    res = cursor.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT *
        FROM db_index
        WHERE scenario = 'scenario_23';
        """
    ).fetchall()

res

# %%
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_zn)) as con:
    cursor = con.cursor()
    res = cursor.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT *
        FROM db_index
        WHERE scenario = 'scenario_23'
            AND run = 3;
        """
    ).fetchall()

res

# %%
# %%time
with contextlib.closing(sqlite3.connect(scratch_index_file_sql_zn)) as con:
    cursor = con.cursor()
    # Surely this can't be the right answer
    cursor.row_factory = lambda _, row: row[0]
    res = cursor.execute(
        """
        SELECT DISTINCT file_id
        FROM db_index
        WHERE scenario = 'scenario_23'
            AND run = 3;
        """
    ).fetchall()

# %% [markdown]
# Check performance in a few different modes.

# %%
db.delete()

# %%
# %%time
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
# %%time
# Notes:
# - I'm surprised the threads block when writing CSV,
#   I would have thought the GIL would be released while writing
# - funny that netCDF explodes
if isinstance(db.backend, netCDFBackend):
    print("Threaded saving causes netCDF to break")
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
# %%time
# Default executors so they can differ, but have to be made for each op
db.save(big_df, groupby=groupby, progress=True, max_workers=max_workers)
print(db.load(progress=True, max_workers=max_workers).shape)
db.delete(progress=True, max_workers=max_workers)

# %%
# %%time
# Serial
db.save(big_df, groupby=groupby, progress=True)
print(db.load(progress=True).shape)
db.delete(progress=True)
