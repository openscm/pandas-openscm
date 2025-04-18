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

## Pandas-OpenSCM v0.4.1 (2025-04-12)

### 🐛 Bug Fixes

- Fixed up [pandas_openscm.index_manipulation.update_levels][].
  It now drops unused levels by default first, to avoid applying the updates to values that aren't being used.
  The same fixes are propagated to [pandas_openscm.accessors.DataFramePandasOpenSCMAccessor.update_index_levels][] and [pandas_openscm.index_manipulation.update_index_levels_func][]. ([#14](https://github.com/openscm/pandas-openscm/pull/14))


## Pandas-OpenSCM v0.4.0 (2025-04-11)

### 🆕 Features

- Added [pandas_openscm.index_manipulation.update_levels][] and the corresponding accessor [pandas_openscm.accessors.DataFramePandasOpenSCMAccessor.update_index_levels][] ([#13](https://github.com/openscm/pandas-openscm/pull/13))


## Pandas-OpenSCM v0.3.3 (2025-03-30)

### 🆕 Features

- - Added a method for converting to long data, see [pandas_openscm.accessors.DataFramePandasOpenSCMAccessor.to_long_data][pandas_openscm.accessors.DataFramePandasOpenSCMAccessor.to_long_data] ([#12](https://github.com/openscm/pandas-openscm/pull/12))


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
