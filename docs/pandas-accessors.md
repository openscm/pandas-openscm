# Pandas accessors

Pandas-OpenSCM also provides a [pandas][] accessor.
For details of the implementation of this pattern, see
[pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

The accessors must be registered before they can be used
(we do this to avoid imports of any of our modules having side effects,
which is a pattern we have had bad experiences with in the past).
This is done with
[register_pandas_accessors][pandas_openscm.accessors.register_pandas_accessors],

By default, the accessors are provided under the "openscm" namespace
and this is how the accessors are documented below.
However, the namespace can be customised when using
[register_pandas_accessors][pandas_openscm.accessors.register_pandas_accessors],
should you wish to use a different namespace for the accessor.

For the avoidance of doubt, in order to register/activate the accessors,
you will need to run something like:

```python
from pandas_openscm.accessors import register_pandas_accessors

# The 'pd.DataFrame.openscm' and 'pd.Series.openscm' namespace
# will not be available at this point.

# Register the accessors
register_pandas_accessors()

# The 'pd.DataFrame.openscm' and 'pd.Series.openscm' namespace
# (or whatever other custom namespace you chose to register)
# will now be available.
```

The full accessor API is documented below.

::: pandas_openscm.accessors.dataframe.PandasDataFrameOpenSCMAccessor
    handler: python_accessors
    options:
        namespace: "pd.DataFrame.openscm"
        show_root_full_path: false
        show_root_heading: true

::: pandas_openscm.accessors.series.PandasSeriesOpenSCMAccessor
    handler: python_accessors
    options:
        namespace: "pd.Series.openscm"
        show_root_full_path: false
        show_root_heading: true
