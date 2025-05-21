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

# %% [markdown]
# # How to use OpenSCMDB
#
# In this notebook we describe how to use the `OpenSCMDB` class.
# This class is useful for writing data to
# and reading data from a database of files.

# %% [markdown]
# ## Imports

# %%
import concurrent.futures
import contextlib
import itertools
import tarfile
import tempfile
import traceback
from functools import partial
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
import pandas_indexing as pix
import tqdm

from pandas_openscm.db import (
    DATA_BACKENDS,
    INDEX_BACKENDS,
    AlreadyInDBError,
    EmptyDBError,
    OpenSCMDB,
)
from pandas_openscm.parallelisation import ParallelOpConfig

# %% [markdown]
# ## Basics
#
# The database is file based.
# We simply write files of data to a directory
# and keep track of where the data is using an index file.
# As a result, every `OpenSCMDB` instance
# needs to know which directory to write into.

# %% [markdown]
# In addition, the instance needs to know
# which file format, i.e. backend,
# to use to write and read the data and the index.
# The available data and index backends are below.
# You can, of course, write your own backend
# that matches the required interface.

# %%
DATA_BACKENDS.options

# %%
INDEX_BACKENDS.options

# %% [markdown]
# Knowing the directory in which to work
# and the back-ends to use, we can initialise a database.

# %%
db = OpenSCMDB(
    db_dir=Path(tempfile.mkdtemp()),
    backend_data=DATA_BACKENDS.get_instance("csv"),
    backend_index=INDEX_BACKENDS.get_instance("csv"),
)
db

# %% [markdown]
# ## Empty databases
#
# If you try and operate on an empty database,
# you will either get an error or nothing will happen.

# %%
# Deleting an empty database does nothing
db.delete()

# %%
# Trying to load an empty database raises an error
try:
    db.load()
except EmptyDBError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# ## Saving data
#
# If we have some data, we can save it to the database.
# In theory, this will work with any data type.
# However, it is optimised to timeseries data
# i.e. data that has a multi-index with metadata values for each timeseries,
# columns which define the time axis
# and values for each timeseries.

# %%
df_timeseries_like = pd.DataFrame(
    np.arange(12).reshape(4, 3),
    columns=[2010, 2015, 2025],
    index=pd.MultiIndex.from_tuples(
        [
            ("scenario_a", "climate_model_a", "Temperature", "K"),
            ("scenario_b", "climate_model_a", "Temperature", "K"),
            ("scenario_b", "climate_model_b", "Temperature", "K"),
            ("scenario_b", "climate_model_b", "Ocean Heat Uptake", "J"),
        ],
        names=["scenario", "climate_model", "variable", "unit"],
    ),
)
df_timeseries_like

# %%
# Saving is then trivial
db.save(df_timeseries_like)

# %% [markdown]
# ## Loading data
#
# ### Metadata
#
# The first thing we might want to load is the metadata in our database.

# %%
# Loading metadata is trivial.
# A MultiIndex is returned by default.
metadata = db.load_metadata()
metadata

# %%
# The MultiIndex can be trivially turned into a DataFrame if you wish
metadata.to_frame(index=False)

# %% [markdown]
# ### Data
#
# Then we can load data.

# %%
# By default, we just load everything
db.load()

# %% [markdown]
# #### Sub-selecting
#
# If we don't want all the data, we can sub-select.

# %%
# For example, this can be done using the previously loaded metadata
db.load(metadata[:2])

# %% [markdown]
# When combined with [pandas-indexing](https://github.com/coroa/pandas-indexing),
# this provides particularly powerful functionality.

# %%
db.load(pix.isin(scenario="scenario_b"))

# %%
db.load(pix.ismatch(variable="Ocean*"))

# %% [markdown]
# ## Deleting
#
# If we wish, we can delete our database with `delete`.

# %%
# The database is not empty to start
db.is_empty

# %%
db.delete()
# Now the database is empty
db.is_empty

# %% [markdown]
# ## Summary so far
#
# The functionality shown up to here is the key functionality.
# If all you need to do is basic saving and loading of data,
# this is all you need.
# If you are looking to support more complex use cases
# (which you probably are, otherwise you wouldn't be here),
# read on.

# %% [markdown]
# ## Advanced topics

# %% [markdown]
# ### Sharing the database
#
# If you need to share a database,
# you can zip it and pass it to someone else.

# %% [markdown]
# We start by putting some data in a database.

# %%
top_level_dir = Path(tempfile.mkdtemp())

# %%
db_start = OpenSCMDB(
    db_dir=top_level_dir / "start",
    backend_data=DATA_BACKENDS.get_instance("csv"),
    backend_index=INDEX_BACKENDS.get_instance("csv"),
)
db_start.save(df_timeseries_like)

# %% [markdown]
# Then we create a gzipped tar archive of our database.

# %%
gzipped = top_level_dir / "db_archive.tar.gz"
db_start.to_gzipped_tar_archive(gzipped)

# %% [markdown]
# To demonstrate that this does not rely on the original data,
# we delete the original database.

# %%
db_start.delete()

# %% [markdown]
# We can inspect the tar file's contents.

# %%
with tarfile.open(gzipped) as tar:
    print(f"{tar.getmembers()=}")

# %% [markdown]
# A new database can be initialised from the gzipped tar archive.

# %%
db_moved = OpenSCMDB.from_gzipped_tar_archive(
    gzipped,
    db_dir=top_level_dir / "moved",
)
db_moved

# %% [markdown]
# As above, we remove the archive
# to demonstrate that there is no reliance on it
# for the following operations.

# %%
gzipped.unlink()

# %% [markdown]
# You can then use this database like normal,
# but now from the new location
# (whether on your machine or someone else's).

# %%
db_moved.load()

# %%
db_moved.load(pix.isin(unit="J"))

# %% [markdown]
# We clean up the files before moving onto the next demonstration.

# %%
db_moved.delete()

# %% [markdown]
# ### Grouping data
#
# For many use cases, it won't make sense to save all your data in a single file.
# You can control how the data is grouped while saving using the arguments shown below.

# %%
df_many_timeseries = pd.DataFrame(
    np.arange(5 * 3 * 4 * 4 * 3).reshape(5 * 3 * 4 * 4, 3),
    columns=[2010, 2015, 2025],
    index=pd.MultiIndex.from_tuples(
        [
            (scenario, variant, climate_model, variable, unit)
            for scenario, variant, climate_model, (variable, unit) in itertools.product(
                ["scenario_a", "scenario_b", "scenario_c", "scenario_d", "scenario_e"],
                ["high", "medium", "low"],
                [
                    "climate_model_a",
                    "climate_model_b",
                    "climate_model_c",
                    "climate_model_d",
                ],
                [
                    ("Temperature", "K"),
                    ("Ocean Heat Uptake", "J"),
                    ("Effective Radiative Forcing", "W / m^2"),
                    ("Warming rate", "K / yr"),
                ],
            )
        ],
        names=["scenario", "variant", "climate_model", "variable", "unit"],
    ),
)
df_many_timeseries

# %%
# For example, we might want to group by climate model and scenario on saving
db.save(df_many_timeseries, groupby=["climate_model", "scenario"])

# %% [markdown]
# ### Parallelisation and progress bars
#
# We provide a variety of ways to control parallelisation during operations
# as well as the display of progress bars.
# The simplest way to control these is via the arguments shown below.

# %%
# Turn on default progress bars
# (note that these don't show up in the rendered docs,
# but will if you actually use the package in a notebook)
db.load(progress=True)

# %%
# Turn on default parallelisation
db.load(max_workers=4)

# %%
# Turn on default progress bars and parallelisation
# (again, note that the progress bars don't show up in the rendered docs,
# but will if you actually use the package in a notebook)
db.load(progress=True, max_workers=4)

# %% [markdown]
# If you want fine-grained control over the behaviour,
# then the operations support receiving `ParallelOpConfig` instances
# to control the parallelisation and progress for different operations.
# An example is shown below.

# %%
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    loaded = db.load(
        parallel_op_config=ParallelOpConfig(
            progress_results=partial(
                tqdm.tqdm, desc="Custom description and non-HTML bar"
            ),
            executor=executor,
            progress_parallel_submission=partial(
                tqdm.tqdm, desc="Custom submission description and non-HTML bar"
            ),
        )
    )

loaded

# %% [markdown]
# ### Overwriting data
#
# By default, it is not possible to overwrite data
# that is already in the database.
# This ensures that you don't accidentally overwrite.

# %%
# Trying to load an empty database raises an error
try:
    db.save(df_many_timeseries)
except AlreadyInDBError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# If you are sure, you can overwrite data as shown below.
# As the progress bars show, this happens in two steps:
#
# 1. the new data is written to disk
# 2. the data that it overwrites is removed

# %%
# A helper to ensure our parallel bars
# show in the rendered docs.
# If you're actually using the package,
# you can just use `progress=True` for the majority of use cases.
save_pbars = dict(
    parallel_op_config_save=ParallelOpConfig(partial(tqdm.tqdm, desc="File saving")),
    parallel_op_config_delete=ParallelOpConfig(
        partial(tqdm.tqdm, desc="File deletion")
    ),
    parallel_op_config_rewrite=ParallelOpConfig(
        partial(tqdm.tqdm, desc="File re-writing")
    ),
)
load_pbar = dict(
    parallel_op_config=ParallelOpConfig(partial(tqdm.tqdm, desc="File loading")),
)

# %%
db.save(df_many_timeseries, allow_overwrite=True, **save_pbars)

# %% [markdown]
# Overwriting works whether you are doing a full overwrite or a partial overwrite.
# However, if you are doing a partial overwrite, the operation can be quite expensive.
# The reason is that you have to load the written data
# and re-write the data that is not being overwritten.
# As a result, considering your data grouping carefully
# will make a big difference in performance
# if you expected to be doing lots of overwrites.

# %%
# Save into a single file
db.save(df_many_timeseries, allow_overwrite=True, **save_pbars)

# %%
# Now overwrite only a selection of the data.
# This will force the existing file to be re-written
# so that the data we wish to keep is not lost,
# but we do get rid of the data we are overwriting.
db.save(df_many_timeseries.iloc[:20, :], allow_overwrite=True, **save_pbars)

# %%
# You can disable the warning as shown below
db.save(
    df_many_timeseries.iloc[-20:, :],
    allow_overwrite=True,
    warn_on_partial_overwrite=False,
    **save_pbars,
)

# %% [markdown]
# ### More on grouping
#
# Now that you have seen some more features,
# it is easier to explain why grouping is so important.
# By default, when we save, we save the data as a single file.

# %%
# Before continuing, clear out the database
db.delete()

# %%
# The progress bar shows that three files are being saved, these are:
# 1. the data (as a single file)
# 2. the index
# 3. the file map
db.save(df_many_timeseries, **save_pbars)

# %% [markdown]
# If we save the data like this, then we have to read all of the data every time,
# even if we only want to get a subset of it.

# %%
# All the data is in one file, which is read in its entirety,
# even though we only want some of the data.
db.load(pix.isin(variant="low"), **load_pbar)

# %% [markdown]
# If you know how you are likely to want to access your data,
# you can give yourself much more fine-grained control.

# %%
# For example, if we know that the variants are our group of interest,
# we should save the database like that.
# As the progress bars show, the data is now grouped into more files
# (3, in fact, with the other two save operations being for the index and file map).
db.save(df_many_timeseries, groupby=["variant"], allow_overwrite=True, **save_pbars)

# %%
# If we only want to load one variant, now we only load data for that variant,
# not all of the data.
db.load(pix.isin(variant="low"), **load_pbar)

# %%
# This is clearer if we load data for two variants, but not all.
# The progress bars show that two files are being loaded.
db.load(pix.isin(variant=["low", "high"]), **load_pbar)

# %%
# If we instead want to load all the temperature data for a given climate model,
# because of the way the data is grouped,
# we have to load all the files
# (you can see this by looking at the progress bar output).
# This would be a good example where,
# if loading a single variable-climate model combination
# at a time is of interest,
# the data should be saved with a grouping that supports
# that access pattern more directly.
db.load(pix.isin(climate_model="climate_model_c", variable="Temperature"), **load_pbar)

# %% [markdown]
# The tradeoff with grouping the data can be that reading more files takes more time
# (although sometimes even then not thanks to parallelisation).
# It's up to you to make the grouping choice that suits your access pattern.

# %% [markdown]
# ### Locking
#
# The database supports locking.
# This ensures that e.g. only one process can write to the database.
# Locking is handled via the `index_file_lock` attribute.
# You can set this at initialisation if you wish.
# If it is not supplied, a default value is used.
#
# The lock is an attribute of the instance.
# In other words, every database instance has its own lock.
# We use [filelock](https://py-filelock.readthedocs.io/en/latest/index.html) by default.
# This is a powerful locking library, but it is important to understand its model.
#
# Of most importance is the fact that
# [the locks are recursive](https://py-filelock.readthedocs.io/en/latest/index.html).
# This means that repeated calls to acquire the same lock will not block.

# %%
with db.index_file_lock.acquire():
    # Even though we just acquired the lock,
    # we can acquire it again because of the recursive behaviour.
    db.index_file_lock.acquire()

# %% [markdown]
# However, if we now try and acquire a lock for a database
# which is working on the same directory, we will find that we can't.

# %%
# Get another db instance
# (the issue doesn't arise with the current instance because of the recursive locking)
db_other_view = OpenSCMDB(
    db_dir=db.db_dir,  # Pointing at the same directory
    backend_data=DATA_BACKENDS.get_instance("csv"),
    backend_index=INDEX_BACKENDS.get_instance("csv"),
)
# The default timeout is inifinity,
# so the below would just block forever by default.
# We make this more sensible here.
db_other_view.index_file_lock.timeout = 0.5

# We can't acquire the lock from this other db instance
try:
    db_other_view.index_file_lock.acquire()
except filelock.Timeout:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# The locking is helpful to avoid putting the database in a corrupt state.
# By default, the lock is acquired for all operations.
# You can override this by providing a different lock to the method.

# %%
# The lock is currently being held
db.index_file_lock.is_locked

# %%
# Despite this, the recursive locks
# mean we can still load data with the db instance
# that is holding the lock
db.load()

# %%
# However, we cannot load data from the other instance
try:
    db_other_view.load()
except filelock.Timeout:
    traceback.print_exc(limit=0, chain=False)

# %%
# Unless we override the lock
db_other_view.load(
    # Providing a nullcontext bypasses the lock.
    # Obviously, only do this if you know what you're doing.
    index_file_lock=contextlib.nullcontext()
)

# %% [markdown]
# For more information on the different ways to acquire the lock,
# see the [filelock docs](https://py-filelock.readthedocs.io/en/latest/index.html).
# If you are interested in locking, we really recommend reading the docs.
# The locking is very powerful, but quite subtle
# so you really have to understand the implementation to get the most out of it.

# %%
# Release the lock fully before moving on
# (if you want to know why this is needed,
# read the filelock docs)
db.index_file_lock.release()
db.index_file_lock.is_locked

# %% [markdown]
# ### OpenSCMDBReader
#
# If you just want to read data,
# then we provide a class optimised to this use case, `OpenSCMDBReader`.
# This holds the index in memory,
# so it does not need to be read from disk every time we wish to load data.
#
# The easiest way to create a reader is from an existing `OpenSCMDB` instance.

# %%
reader = db.create_reader()
# the index is in memory
reader.db_index

# %% [markdown]
# The reader's `load` method is basically identical to that of `OpenSCMDB`,
# it just has to do less work because it doesn't need to read the index.

# %%
# Having the index in memory can make data reading faster
# (in the case of a large index).
reader.load(pix.isin(unit="K", variant="medium") & pix.ismatch(scenario="*_c"))

# %% [markdown]
# By default, the reader is given its own lock.
# This allows us to use the reader to lock the database.

# %%
reader.lock.acquire()

# %%
# Now we can't perform operations with any other view into the db,
# so we can gurantee our reader's safety.
try:
    db_other_view.load()
except filelock.Timeout:
    traceback.print_exc(limit=0, chain=False)

# %%
# If we release the lock, we can load again
reader.lock.release()
db_other_view.load()

# %% [markdown]
# The reader can also be used as a context manager,
# in which the lock is automatically acquired and released.

# %%
with reader:
    # The lock is held by the reader here
    # so we can load data via the reader
    reader.load()
    # but we can't e.g. save data via another view into the database
    try:
        db_other_view.save(df_timeseries_like)
    except filelock.Timeout:
        traceback.print_exc(limit=0, chain=False)

# Outside the context block, the lock is released by the reader
# so we can use other views for operations again.
db_other_view.load()

# %% [markdown]
# If you aren't worried about the reader having a lock,
# you can simply disable it when creating the reader.

# %%
reader_no_lock = db.create_reader(lock=False)
reader_no_lock.lock is None

# %% [markdown]
# ### Summary
#
# We believe that OpenSCMDB provides a powerful way for saving timeseries data.
# However, it is in a work progress.
# If there is a feature you would like or a bug or anything else,
# please [raise an issue](https://github.com/openscm/pandas-openscm/issues).
