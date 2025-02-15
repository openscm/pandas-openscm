"""
[Feather](https://arrow.apache.org/docs/python/feather.html) backend
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from attrs import define


@define
class FeatherBackend:
    """
    Feather-backend for our database

    For details on feather, see https://arrow.apache.org/docs/python/feather.html
    """

    ext: str = ".feather"
    """
    Extension to use with files saved by this backend.
    """

    @property
    def preserves_index(self) -> Literal[True]:
        """
        Whether this backend preserves the index
        """
        return True

    @staticmethod
    def load_data_file(data_file: Path) -> pd.DataFrame:
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
        return pd.read_feather(file_map_file)

    @staticmethod
    def load_index(index_file: Path) -> pd.DataFrame:
        """
        Load the database's index

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
    def save_data(data: pd.DataFrame, data_file: Path) -> None:
        """
        Save an individual data file to the database

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

    def save_index_and_file_map(
        self,
        index: pd.DataFrame,
        index_file: Path,
        file_map: pd.Series[Path],  # type: ignore # pandas confused about what it supports
        file_map_file: Path,
    ) -> None:
        """
        Save the database

        Parameters
        ----------
        index
            Index file to save

        index_file
            File in which to save the index

        file_map
            File map to save

        file_map_file
            File in which to save the file map
        """
        index.to_feather(index_file)

        # Feather doesn't support
        # (see https://pandas.pydata.org/docs/user_guide/io.html#feather):
        # - writing indexes
        file_map_write = file_map.reset_index()
        # - writing non-native types (e.g. Path)
        file_map_write["file_path"] = file_map_write["file_path"].astype(str)
        file_map_write.to_feather(file_map_file)
