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
import tempfile
from pathlib import Path

import numpy as np

from pandas_openscm.db import FeatherBackend, OpenSCMDB
from pandas_openscm.testing import create_test_df

# %%
units = ["K", "W", "yr"]
big_df = create_test_df(
    n_scenarios=500,
    n_runs=60,
    timepoints=np.arange(1750.0, 2100.0 + 1.0),
    variables=[(f"variable_{i}", units[i % len(units)]) for i in range(4)],
)
big_df

# %%
db_dir = Path(tempfile.mkdtemp())
db = OpenSCMDB(db_dir=db_dir, backend=FeatherBackend())

# %%
import tqdm.autonotebook as tqdman

for _, svdf in tqdman.tqdm(big_df.groupby(["scenario", "variable"])):
    db.save(svdf)

# %%
import concurrent.futures
from functools import partial

# %%
db.load(progress_results=True)

# %%
with concurrent.futures.ThreadPoolExecutor() as executor:
    db.load(
        progress_parallel_submission=partial(tqdman.tqdm, ncols=400),
        progress_results=partial(tqdman.tqdm, ncols=500),
        executor=executor,
    )

# %%
# Overhead of spinning up new processes is the killer here.
# Hence threads are a good default.
with concurrent.futures.ProcessPoolExecutor() as executor:
    db.load(progress_results=True, progress_parallel_submission=True, executor=executor)
