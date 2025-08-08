# Pandas accessors

Pandas-OpenSCM also provides [pandas][] accessors.
For details of the implementation of this pattern, see
[pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

The accessors must be registered before they can be used
(we do this to avoid imports of any of our modules having side effects,
which is a pattern we have had bad experiences with in the past).
This is done with
[register_pandas_accessors][pandas_openscm.accessors.register_pandas_accessors],

By default, the accessors are provided under the "openscm" namespace.
However, the namespace can be customised when using
[register_pandas_accessors][pandas_openscm.accessors.register_pandas_accessors],
should you wish to use a different namespace for the accessor.

For the avoidance of doubt, in order to register/activate the accessors,
you will need to run something like:

```python
from pandas_openscm.accessors import register_pandas_accessors

# The 'pd.DataFrame.openscm' and 'pd.Series.openscm' namespaces
# will not be available at this point.

# Register the accessors
register_pandas_accessors()

# The 'pd.DataFrame.openscm' and 'pd.Series.openscm' namespaces
# will now be available.
# I.e. you could now do something like
df = pd.DataFrame(
    [
        [1.1, 0.8, 1.2],
        [2.1, np.nan, 8.4],
    ],
    columns=[2010.0, 2015.0, 2025.0],
    index=pd.MultiIndex.from_tuples(
        [
            ("sa", "v2", "W"),
            ("sb", "v2", "W"),
        ],
        names=["scenario", "variable", "unit"],
    ),
)

# Use pandas-openscm's functionality via the registered accessors.
df.openscm.to_long_data()

# If you want to register the accessors under a custom namespace instead,
# use something like the below instead
register_pandas_accessors(namespace="my_custom_namespace")

# Doing it this way will make the custom namespace available under
# 'pd.DataFrame.my_custom_namespace' and 'pd.Series.my_custom_namespace'.
```

The full accessor APIs are documented at
[pandas_openscm.accessors.dataframe.PandasDataFrameOpenSCMAccessor][]
and [pandas_openscm.accessors.series.PandasSeriesOpenSCMAccessor][].
