# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## Pandas-OpenSCM v0.8.0 (2025-08-05)

### ⚠️ Breaking Changes

- - Renamed pandas_openscm.register_pandas_accessor to [pandas_openscm.register_pandas_accessors][] (with a trailing 's') as accessors are now also registered for [pandas Series][pandas.Series]
  - Renamed pandas_openscm.accessors.DataFramePandasOpenSCMAccessor to [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor][]
  - Renamed `df_unit_level` to `unit_level` in [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.convert_unit_like][]
  - Renamed `df` to `pobj` in [pandas_openscm.index_manipulation.set_index_levels_func][], [pandas_openscm.unit_conversion.convert_unit_from_target_series][], [pandas_openscm.unit_conversion.convert_unit][] and [pandas_openscm.unit_conversion.convert_unit_like][]

  ([#24](https://github.com/openscm/pandas-openscm/pull/24))

### 🆕 Features

- Added some accessors for [pandas Series][pandas.Series] via [pandas_openscm.accessors.PandasSeriesOpenSCMAccessor][]. Note that this is not feature complete yet, tracking in [#25](https://github.com/openscm/pandas-openscm/issues/25) ([#24](https://github.com/openscm/pandas-openscm/pull/24))

### 🎉 Improvements

- [pandas_openscm.index_manipulation.set_index_levels_func][], [pandas_openscm.unit_conversion.convert_unit_from_target_series][], [pandas_openscm.unit_conversion.convert_unit][] and [pandas_openscm.unit_conversion.convert_unit_like][] now explicitly support [pd.Series][pandas.Series] ([#24](https://github.com/openscm/pandas-openscm/pull/24))

### 🔧 Trivial/Internal Changes

- [#24](https://github.com/openscm/pandas-openscm/pull/24)


## Pandas-OpenSCM v0.7.0 (2025-07-24)

### 🆕 Features

- - Added unit conversion APIs: [pandas_openscm.unit_conversion.convert_unit] and [pandas_openscm.unit_conversion.convert_unit_like] and the corresponding accessors [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.convert_unit] and [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.convert_unit_like]
  - Added the helper: [pandas_openscm.index_manipulation.ensure_is_multiindex] and the corresponding accessors [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.ensure_index_is_multiindex] and [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.eiim]

  ([#23](https://github.com/openscm/pandas-openscm/pull/23))


## Pandas-OpenSCM v0.6.0 (2025-07-16)

### ⚠️ Breaking Changes

- Updated minimum numpy version to 1.26.0, the earliest that is not in end-of-life. Fixed the numpy pin for Python 3.13 to >=2.1.0, the first numpy version which supported Python 3.13. ([#21](https://github.com/openscm/pandas-openscm/pull/21))

### 🔧 Trivial/Internal Changes

- [#21](https://github.com/openscm/pandas-openscm/pull/21), [#22](https://github.com/openscm/pandas-openscm/pull/22)


## Pandas-OpenSCM v0.5.1 (2025-05-23)

### 🆕 Features

- Added [pandas_openscm.index_manipulation.set_levels][] and the corresponding accessor [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.set_index_levels][] ([#18](https://github.com/openscm/pandas-openscm/pull/18))


## Pandas-OpenSCM v0.5.0 (2025-05-10)

### ⚠️ Breaking Changes

- - Required `db_dir` to be passed when initialising [pandas_openscm.db.reader.OpenSCMDBReader][]. This is required to support portable databases
  - Renamed `out_column_type` to `out_columns_type` in [pandas_openscm.io.load_timeseries_csv][] for consistency with the rest of the API
  - Bumped the minimum supported version of [filelock](https://py-filelock.readthedocs.io/) to 3.12.3, as only this version handles automatic creation of directories for the lock

  ([#19](https://github.com/openscm/pandas-openscm/pull/19))

### 🆕 Features

- - Made the database portable by only storing relative paths in the file map. This allows the database to be converted to an archive with [pandas_openscm.db.OpenSCMDB.to_gzipped_tar_archive][] and then unpacked elsewhere with [pandas_openscm.db.OpenSCMDB.from_gzipped_tar_archive][]
  - Added [pandas_openscm.db.path_handling][] to clarify how we handle paths internally to support portability
  - Added support for specifying the name of the output columns via [pandas_openscm.db.OpenSCMDB.load][], [pandas_openscm.db.reader.OpenSCMDBReader.load][] and [pandas_openscm.io.load_timeseries_csv][]

  ([#19](https://github.com/openscm/pandas-openscm/pull/19))

### 🎉 Improvements

- - Added the explicit [pandas_openscm.db.backends][] module to handle the backends we support more clearly
  - Added [pandas_openscm.db.backends.DataBackendOptions.guess_backend][] and [pandas_openscm.db.backends.IndexBackendOptions.guess_backend][] to allow for move convenient inference of the backend to use with different files

  ([#19](https://github.com/openscm/pandas-openscm/pull/19))

### 🔧 Trivial/Internal Changes

- [#19](https://github.com/openscm/pandas-openscm/pull/19)


## Pandas-OpenSCM v0.4.2 (2025-05-05)

### 🆕 Features

- Add compare_close function to compare two dataframes. ([#16](https://github.com/openscm/pandas-openscm/pull/16))
- Added [pandas_openscm.index_manipulation.update_levels_from_other][] and the corresponding accessor [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.update_index_levels_from_other][] ([#17](https://github.com/openscm/pandas-openscm/pull/17))


## Pandas-OpenSCM v0.4.1 (2025-04-12)

### 🐛 Bug Fixes

- Fixed up [pandas_openscm.index_manipulation.update_levels][].
  It now drops unused levels by default first, to avoid applying the updates to values that aren't being used.
  The same fixes are propagated to [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.update_index_levels][] and [pandas_openscm.index_manipulation.update_index_levels_func][]. ([#14](https://github.com/openscm/pandas-openscm/pull/14))


## Pandas-OpenSCM v0.4.0 (2025-04-11)

### 🆕 Features

- Added [pandas_openscm.index_manipulation.update_levels][] and the corresponding accessor [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.update_index_levels][] ([#13](https://github.com/openscm/pandas-openscm/pull/13))


## Pandas-OpenSCM v0.3.3 (2025-03-30)

### 🆕 Features

- - Added a method for converting to long data, see [pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.to_long_data][pandas_openscm.accessors.PandasDataFrameOpenSCMAccessor.to_long_data] ([#12](https://github.com/openscm/pandas-openscm/pull/12))


## Pandas-OpenSCM v0.3.2 (2025-03-27)

### 🆕 Features

- Added release of the package on conda ([#9](https://github.com/openscm/pandas-openscm/pull/9))
- Added a basic de-serialisation, i.e. IO, module [pandas_openscm.io][pandas_openscmio] ([#11](https://github.com/openscm/pandas-openscm/pull/11))

### 🔧 Trivial/Internal Changes

- [#9](https://github.com/openscm/pandas-openscm/pull/9), [#12](https://github.com/openscm/pandas-openscm/pull/12)


## Pandas-OpenSCM v0.3.1 (2025-03-23)

### 🔧 Trivial/Internal Changes

- [#8](https://github.com/openscm/pandas-openscm/pull/8)


## Pandas-OpenSCM v0.3.0 (2025-03-23)

### 🆕 Features

- - Added the plotting module, [pandas_openscm.plotting][pandas_openscmplotting] and associated accessors.
  - Added the grouping module, [pandas_openscm.grouping][pandas_openscmgrouping] and associated accessors.

  ([#7](https://github.com/openscm/pandas-openscm/pull/7))

### 🐛 Bug Fixes

- Added LICENCE ([#5](https://github.com/openscm/pandas-openscm/pull/5))

### 🔧 Trivial/Internal Changes

- [#3](https://github.com/openscm/pandas-openscm/pull/3), [#4](https://github.com/openscm/pandas-openscm/pull/4), [#6](https://github.com/openscm/pandas-openscm/pull/6)


## Pandas-OpenSCM v0.2.0 (2025-03-17)

### 🆕 Features

- Added the database module, see [pandas_openscm.db][pandas_openscmdb] ([#2](https://github.com/openscm/pandas-openscm/pull/2))


## Pandas-OpenSCM v0.1.0 (2025-02-09)

No significant changes.
