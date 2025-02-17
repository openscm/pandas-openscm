"""
netCDF backend
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from attrs import define

from pandas_openscm.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import xarray as xr


@define
class netCDFDataBackend:
    """
    netCDF data backend
    """

    ext: str = ".nc"
    """
    Extension to use with files saved by this backend.
    """

    timeseries_dim: str = "ts_id"
    """
    Name of the timeseries dimension in serialised output
    """

    @property
    def preserves_index(self) -> Literal[True]:
        """
        Whether this backend preserves the index of data upon (de-)serialisation
        """
        return True

    def load_data(self, data_file: Path) -> pd.DataFrame:
        """
        Load a data file

        Parameters
        ----------
        data_file
            File from which to load the data

        Returns
        -------
        :
            Loaded data
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "netCDFBackend.load_data", requirement="xarray"
            ) from exc

        raw = xr.load_dataset(data_file)

        data: pd.DataFrame = raw["values"].to_pandas()  # type: ignore
        index = metadata_xr_to_df(raw)
        index_concat = index.loc[raw[self.timeseries_dim].values]

        res = pd.concat([index_concat, data], axis="columns").set_index(
            index.columns.to_list()
        )

        return res

    def save_data(self, data: pd.DataFrame, data_file: Path) -> None:
        """
        Save data to disk

        Parameters
        ----------
        data
            Data to save

        data_file
            File in which to save the data
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "netCDFBackend.save_data", requirement="xarray"
            ) from exc

        # Resetting the index will also give each timeseries a unique ID
        data_rs = data.reset_index()
        timeseries_coord_info = {self.timeseries_dim: data_rs.index.values}
        time_coord_info = {"time": data.columns}
        data_index_xr = metadata_df_to_xr(
            data_rs[data.index.names],
            timeseries_id_coord=xr.Coordinates(timeseries_coord_info),
            timeseries_dim=self.timeseries_dim,
        )
        data_values_xr = xr.DataArray(
            data,
            dims=[self.timeseries_dim, "time"],
            coords=xr.Coordinates(timeseries_coord_info | time_coord_info),
        )
        data_xr = xr.merge([data_index_xr, data_values_xr.to_dataset(name="values")])
        data_xr.to_netcdf(data_file)


@define
class netCDFIndexBackend:
    """
    netCDF index backend
    """

    ext: str = ".nc"
    """
    Extension to use with files saved by this backend.
    """

    timeseries_dim: str = "ts_id"
    """
    Name of the timeseries dimension in serialised output
    """

    @property
    def preserves_index(self) -> Literal[True]:
        """
        Whether this backend preserves the `pd.MultiIndex` upon (de-)serialisation
        """
        return True

    @staticmethod
    def load_file_map(file_map_file: Path) -> pd.DataFrame:
        """
        Load the database's file map

        Parameters
        ----------
        file_map_file
            File from which to load the file map

        Returns
        -------
        :
            Loaded file map
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "netCDFBackend.load_file_map", requirement="xarray"
            ) from exc

        res = xr.load_dataset(file_map_file).to_pandas()
        if isinstance(res, pd.Series):
            raise TypeError

        return res

    @staticmethod
    def load_index(index_file: Path) -> pd.DataFrame:
        """
        Load the index

        Parameters
        ----------
        index_file
            File from which to load the index

        Returns
        -------
        :
            Loaded index
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "netCDFBackend.load_index", requirement="xarray"
            ) from exc

        raw = xr.load_dataset(index_file)

        intermediate = metadata_xr_to_df(raw)
        res = intermediate.set_index(
            intermediate.columns.difference(["file_id"]).to_list()
        )

        return res

    @staticmethod
    def save_file_map(
        file_map: pd.Series[Path],  # type: ignore # pandas confused about what it supports
        file_map_file: Path,
    ) -> None:
        """
        Save the file map to disk

        Parameters
        ----------
        file_map
            File map to save

        file_map_file
            File in which to save the file map
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "netCDFBackend.save_file_map", requirement="xarray"
            ) from exc

        file_map_xr = xr.DataArray.from_series(file_map.astype(str))
        file_map_xr.to_netcdf(file_map_file)

    def save_index(
        self,
        index: pd.DataFrame,
        index_file: Path,
    ) -> None:
        """
        Save the index to disk

        Parameters
        ----------
        index
            Index to save

        index_file
            File in which to save the index
        """
        # Use a different name because the timeseries IDs in the index
        # won't necessarily line up with those in the file(s).
        # This should not matter for users, who never see them side-by-side,
        # but just in case.
        index_xr = metadata_df_to_xr(
            # Have to reset the index so we can serialise to disk
            index.reset_index(),
            timeseries_dim=f"{self.timeseries_dim}_index",
        )
        index_xr.to_netcdf(index_file)


def metadata_df_to_xr(
    metadata: pd.DataFrame,
    timeseries_id_coord: xr.Coordinates | None = None,
    timeseries_dim: str = "ts_id",
) -> xr.Dataset:
    """
    Convert metadata to a format that can be used with xarray

    This assumes that each row is the metadata for an individual timeseries
    and that the index of `metadata` should be used for the unique ID
    of each timeseries.

    This avoids the metadata creating super sparse arrays,
    at the expense of being harder to work with if directly used with xarray.
    If you need to work with super sparse data
    and want to work with it in a lower-dimensional data structure
    to avoid sparsity issues, our advice would be to use pandas.
    For docs on pandas support for higher dimensional data,
    see https://pandas.pydata.org/docs/user_guide/advanced.html#hierarchical-indexing-multiindex.

    Parameters
    ----------
    metadata
        Metadata to convert

    timeseries_id_coord
        Co-ordinate to use for the timeseries ID.

        If not supplied, this will be created from `metadata.index`.

    timeseries_dim
        Name of the dimension to use to label timeseries in the output

    Returns
    -------
    :
        Metadata, converted to a 'flat' xarray object.

    Raises
    ------
    MissingOptionalDependencyError
        [xarray][] is not installed.

    AssertionError
        `timeseries_id_coord` is `None`
        and the index of `metadata` is not already unique.

    See Also
    --------
    metadata_xr_to_df
    """
    # TODO: doctests/examples
    try:
        import xarray as xr
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "metadata_to_xr", requirement="xarray"
        ) from exc

    if timeseries_id_coord is None:
        if metadata.index.duplicated().any():
            dups = metadata.index[metadata.index.duplicated(keep=False)]
            msg = "Index values should be unique. " f"Received these duplicates: {dups}"
            raise AssertionError(msg)

    metadata_columns = metadata.columns.tolist()
    index_darrays = {}
    for metadata_col in metadata_columns:
        metadata_col_unique_vals = metadata[metadata_col].unique().tolist()
        metadata_col_map = {}
        metadata_col_mapped_values = [0] * len(metadata_col_unique_vals)
        for i, v in enumerate(metadata_col_unique_vals):
            metadata_col_map[v] = i
            metadata_col_mapped_values[i] = i

        metadata_map_xr = xr.DataArray(
            metadata_col_unique_vals,
            # make naming injectable (?)
            dims=[f"{metadata_col}_map"],
            coords={f"{metadata_col}_map": metadata_col_mapped_values},
        )
        metadata_int_xr = xr.DataArray(
            # Not sure if converting to category first then mapping
            # is faster here or not, would have to profile.
            metadata[metadata_col].map(metadata_col_map).astype(int),
            dims=[timeseries_dim],
            coords=timeseries_id_coord,
        )

        index_darrays[metadata_col] = metadata_map_xr
        # make naming injectable (?)
        index_darrays[f"{metadata_col}_int"] = metadata_int_xr

    index_xr = xr.Dataset(index_darrays)

    return index_xr


def metadata_xr_to_df(
    metadata: xr.Dataset,
    category_index: bool = False,
) -> pd.DataFrame:
    """
    Convert metadata in [xarray][] form to [pandas][]

    Parameters
    ----------
    metadata
        Metadata to convert

    category_index
        Should the index be returned as category type?

    Returns
    -------
    :
        [pandas][] form of the metadata

    See Also
    --------
    metadata_df_to_xr
    """
    # TODO: doctests/examples
    # make naming injectable (?)
    metadata_columns = [
        str(v).replace("_map", "") for v in metadata.coords if str(v).endswith("_map")
    ]

    index_cols = {}
    for metadata_col in metadata_columns:
        # make naming injectable (?)
        metadata_col_series_int = metadata[f"{metadata_col}_int"].to_pandas()
        if category_index:
            metadata_col_series_int = metadata_col_series_int.astype("category")

        metadata_col_map: pd.Series[str] = metadata[metadata_col].to_pandas()  # type: ignore # xarray confused

        index_cols[metadata_col] = metadata_col_series_int.map(metadata_col_map)  # type: ignore # pandas map type hint confused

    res = pd.DataFrame(index_cols)

    return res
