"""
CSV backend
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from attrs import define


@define
class CSVDataBackend:
    """
    CSV data backend
    """

    ext: str = ".csv"
    """
    Extension to use with files saved by this backend.
    """

    @property
    def preserves_index(self) -> Literal[False]:
        """
        Whether this backend preserves the index of data upon (de-)serialisation
        """
        return False

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
        return pd.read_csv(data_file)

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
        data.to_csv(data_file)


@define
class CSVIndexBackend:
    """
    CSV index backend
    """

    ext: str = ".csv"
    """
    Extension to use with files saved by this backend.
    """

    @property
    def preserves_index(self) -> Literal[False]:
        """
        Whether this backend preserves the `pd.MultiIndex` upon (de-)serialisation
        """
        return False

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
        return pd.read_csv(file_map_file)

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
        return pd.read_csv(index_file)

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
        file_map.to_csv(file_map_file)

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
        index.to_csv(index_file)
