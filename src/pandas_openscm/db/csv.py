"""
CSV-backend
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from attrs import define


@define
class CSVBackend:
    """
    CSV-backend for our database
    """

    ext: str = ".csv"
    """
    Extension to use with files saved by this backend.
    """

    @property
    def preserves_index(self) -> Literal[False]:
        """
        Whether this backend preserves the index

        (Hint, it doesn't)
        """
        return False

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
        return pd.read_csv(data_file)

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
        return pd.read_csv(file_map_file)

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
        return pd.read_csv(index_file)

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
        data.to_csv(data_file)

    def save_file_map(
        self,
        file_map: pd.Series[Path],  # type: ignore # pandas confused about what it supports
        file_map_file: Path,
    ) -> None:
        """
        Save the file map

        Parameters
        ----------
        file_map
            File map to save

        file_map_file
            File in which to save the file map
        """
        file_map.to_csv(file_map_file)

    def save_index(
        self,
        index: pd.DataFrame,
        index_file: Path,
    ) -> None:
        """
        Save the index

        Parameters
        ----------
        index
            Index to save

        index_file
            File in which to save the index
        """
        index.to_csv(index_file)
