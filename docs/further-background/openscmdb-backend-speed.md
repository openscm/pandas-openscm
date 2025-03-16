# OpenSCMDB back-end speed

Here we present an analysis of the speed of different back-ends available to use with [OpenSCMDB][pandas_openscmdbopenscm_db].
For each back-end, we show how fast it is to read and write a collection of timeseries.
We consider use cases that mirror the use cases we have.
If you have a use case that you would like us to include in this analysis,
please [raise an issue](https://github.com/openscm/pandas-openscm/issues/new/choose).

## Run-time conditions

---8<--- "docs/further-background/openscmdb-backend-speed/high-level-info.txt"

## Simple climate model full output

Here we consider the case of output that mirrors the output produced by simple climate models.
We have a number of scenarios and variables, each of which has hundreds of ensemble members.
The output also spans a relatively long time period.

---8<--- "docs/further-background/openscmdb-backend-speed/full-scm-output.txt"

### Storing the index as a category type

Storing the index as a category type
(full docs [here](https://pandas.pydata.org/docs/user_guide/categorical.html))
can significantly reduce the memory use of the data,
the write times and the size on disk.
If you are optimising for performance, this step is worth considering,
especially with back-ends that are able to account for the category type.

---8<--- "docs/further-background/openscmdb-backend-speed/full-scm-output-category-index.txt"

## Simple climate model future quantile output

Here we consider the case of output that mirrors processed output produced from simple climate models.
We have a number of scenarios and variables, each of which has been processed to a few quantiles.
The output is restricted to the future time period.

---8<--- "docs/further-background/openscmdb-backend-speed/scm-future-quantiles-output.txt"


### Storing the index as a category type

As above, we also present results with the index as a category type.

---8<--- "docs/further-background/openscmdb-backend-speed/scm-future-quantiles-output-category-index.txt"
