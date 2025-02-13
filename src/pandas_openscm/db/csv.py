"""
CSV-backend
"""

from __future__ import annotations

from pathlib import Path

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

    def save_database(  # noqa: PLR0913
        self,
        index: pd.DataFrame,
        index_file: Path,
        file_map: pd.Series[Path],  # type: ignore # pandas confused about what it supports
        file_map_file: Path,
        data: pd.DataFrame,
        data_file: Path,
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

        data
            Data to save

        data_file
            File in which to save the data
        """
        index.to_csv(index_file, index=False)
        file_map.reset_index().to_csv(file_map_file, index=False)
        self.save_data(data, data_file)
