# OpenSCMDB back-end speed

Here we present an analysis of the speed of different back-ends available to use with [OpenSCMDB][pandas_openscmdbopenscm_db].
For each back-end, we show how fast it is to read and write a collection of timeseries.
We consider use cases that mirror the use cases we have.
If you have a use case that you would like us to include in this analysis,
please [raise an issue](https://github.com/openscm/pandas-openscm/issues/new/choose).

## Test information

---8<--- "docs/further-background/high-level-info.txt"

## Simple climate model full output

Here we consider the case of output that mirrors the output produced by simple climate models.
We have a number of scenarios and variables, each of which has hundreds of ensemble members.
The output also spans a relatively long time period.

- n scenarios
- n variables
- n timeseries
- n time points
- data size in memory

- rows: back-ends
- columns: operations (write, write parallel, load, load parallel, delete)

For comparison, we include the time it takes to do things similar to the above
directly via pandas (i.e. bypassing the database processing).
These numbers help to identify where the database 'stuff'
introduces significant overhead.

- rows: back-ends
- columns: operations (save index, save file map, save data all in one, save data groupby (not parallel))

### Storing the index as a category type

Storing the index as a category type
(full docs [here](https://pandas.pydata.org/docs/user_guide/categorical.html))
can significantly reduce the memory use of the data and the write times.
If you are optimising for performance, this step is worth considering,
especially with back-ends that are able to account for the category type.

- rows: back-ends
- columns: operations and whether category or not

## Simple climate model future quantile output

Here we consider the case of output that mirrors processed output produced from simple climate models.
We have a number of scenarios and variables, each of which has been processed to a few quantiles.
The output is restricted to the future time period.

- n scenarios
- n variables
- n timeseries
- n time points
- data size in memory

- rows: back-ends
- columns: operations (write, write parallel, load, load parallel, delete)

For comparison, we include the time it takes to do things similar to the above
directly via pandas (i.e. bypassing the database processing).
These numbers help to identify where the database 'stuff'
introduces significant overhead.

- rows: back-ends
- columns: operations (save index, save file map, save data all in one, save data groupby (not parallel))

### Storing the index as a category type

Storing the index as a category type
(full docs [here](https://pandas.pydata.org/docs/user_guide/categorical.html))
can significantly reduce the memory use of the data and the write times.
If you are optimising for performance, this step is worth considering,
especially with back-ends that are able to account for the category type.

- rows: back-ends
- columns: operations and whether category or not
