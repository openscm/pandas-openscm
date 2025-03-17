"""
[Feather](https://arrow.apache.org/docs/python/feather.html) backend
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from attrs import define


@define
class FeatherDataBackend:
    """
    Feather data backend

    For details on feather, see https://arrow.apache.org/docs/python/feather.html
    """

    ext: str = ".feather"
    """
    Extension to use with files saved by this backend.
    """

    @property
    def preserves_index(self) -> Literal[True]:
        """
        Whether this backend preserves the index of data upon (de-)serialisation
        """
        return True

    @staticmethod
    def load_data(data_file: Path) -> pd.DataFrame:
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
        return pd.read_feather(data_file)

    @staticmethod
    def save_data(data: pd.DataFrame, data_file: Path) -> None:
        """
        Save data to disk

        Parameters
        ----------
        data
            Data to save

        data_file
            File in which to save the data
        """
        # The docs say that feather doesn't support writing indexes
        # # (see https://pandas.pydata.org/docs/user_guide/io.html#feather).
        # However, it seems to have no issue writing our multi-indexes.
        # Hence the implementation below
        data.to_feather(data_file)


@define
class FeatherIndexBackend:
    """
    Feather index backend

    For details on feather, see https://arrow.apache.org/docs/python/feather.html
    """

    ext: str = ".feather"
    """
    Extension to use with files saved by this backend.
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
        Load the file map

        Parameters
        ----------
        file_map_file
            File from which to load the file map

        Returns
        -------
        :
            Loaded file map
        """
        return pd.read_feather(file_map_file)

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
        return pd.read_feather(index_file)

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
        # Feather doesn't support writing non-native types
        # (see https://pandas.pydata.org/docs/user_guide/io.html#feather).
        # The docs say that feather doesn't support writing indexes
        # # (see https://pandas.pydata.org/docs/user_guide/io.html#feather).
        # However, it seems to have no issue writing this index.
        # Hence the implementation below
        file_map_write = file_map.astype(str)
        file_map_write.to_frame().to_feather(file_map_file)

    @staticmethod
    def save_index(
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
        index.to_feather(index_file)
