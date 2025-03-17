"""
Interfaces used throughout the db (database) module
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class OpenSCMDBIndexBackend(Protocol):
    """
    Backend for (de-)serialising the index (and file map)

    Designed to be used with [OpenSCMDB][(m)]
    """

    ext: str
    """
    Extension to use with index files saved by this backend.
    """

    preserves_index: bool
    """
    Whether this backend preserves the `pd.MultiIndex` upon (de-)serialisation
    """

    @staticmethod
    def load_file_map(file_map_file: Path) -> pd.DataFrame:
        """
        Load the file map

        This is a low-level method
        that just handles the specifics of loading the index from disk.
        Working out the path from which to load the file map
        should happen in higher-level functions.

        Parameters
        ----------
        file_map_file
            File from which to load the file map

        Returns
        -------
        :
            Loaded file map

        Notes
        -----
        This returns a [pd.DataFrame][pandas.DataFrame].
        It is up to the user to cast this to a [pd.Series][pandas.Series]
        if they wish.
        """

    @staticmethod
    def load_index(index_file: Path) -> pd.DataFrame:
        """
        Load the index

        This is a low-level method
        that just handles the specifics of loading the index from disk.
        Working out the path from which to load the index
        should happen in higher-level functions.

        Parameters
        ----------
        index_file
            File from which to load the index

        Returns
        -------
        :
            Loaded index

        Notes
        -----
        This just loads the index directly from disk.
        If the index had a `pd.MultiIndex` when it was saved,
        this may or not be restored.
        It is up to the user
        to decide whether to do any `pd.MultiIndex` restoration or not,
        based on their use case and the value of `self.preserves_index`.
        We do not make this choice as converting back to a
        `pd.MultiIndex` can be a very expensive operation,
        and we want to give the user control over any such optimisations.
        """

    def save_file_map(
        self,
        file_map: pd.Series[Path],  # type: ignore # pandas confused about what it supports
        file_map_file: Path,
    ) -> None:
        """
        Save the file map to disk

        This is a low-level method
        that just handles the specifics of serialising the file map to disk.
        Working out what to save and in what path
        should happen in higher-level functions.

        Parameters
        ----------
        file_map
            File map to save

        file_map_file
            File in which to save the file map
        """

    def save_index(
        self,
        index: pd.DataFrame,
        index_file: Path,
    ) -> None:
        """
        Save the index to disk

        This is a low-level method
        that just handles the specifics of serialising the index to disk.
        Working out what to save and in what path
        should happen in higher-level functions.

        Parameters
        ----------
        index
            Index to save

        index_file
            File in which to save the index
        """


@runtime_checkable
class OpenSCMDBDataBackend(Protocol):
    """
    Backend for (de-)serialising data

    Designed to be used with [OpenSCMDB][(m)]
    """

    ext: str
    """
    Extension to use with data files saved by this backend.
    """

    preserves_index: bool
    """
    Whether this backend preserves the index of data upon (de-)serialisation
    """

    @staticmethod
    def load_data(data_file: Path) -> pd.DataFrame:
        """
        Load a data file

        This is a low-level method
        that just handles the specifics of loading the data from disk.
        Working out the path from which to load the data
        should happen in higher-level functions.

        Parameters
        ----------
        data_file
            File from which to load the data

        Returns
        -------
        :
            Loaded data

        Notes
        -----
        This just loads the data directly from disk.
        If the data had a `pd.MultiIndex` when it was saved,
        this may or not be restored.
        It is up to the user
        to decide whether to do any `pd.MultiIndex` restoration or not,
        based on their use case and the value of `self.preserves_index`.
        We do not make this choice as converting back to a
        `pd.MultiIndex` can be a very expensive operation,
        and we want to give the user control over any such optimisations.
        """

    @staticmethod
    def save_data(
        data: pd.DataFrame,
        data_file: Path,
    ) -> None:
        """
        Save data to disk

        This is a low-level method
        that just handles the specifics of serialising the data to disk.
        Working out what to save and in what path
        should happen in higher-level functions.

        Parameters
        ----------
        data
            Data to save

        data_file
            File in which to save the data
        """
