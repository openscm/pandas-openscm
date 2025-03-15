"""
In-memory backend

Not very useful in practice, but helpful for testing
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from attrs import define


@define
class InMemoryDataBackend:
    """
    In-memory data backend
    """

    ext: str = ".in-mem"
    """
    Extension to use with files saved by this backend.
    """

    data: dict[str, pd.DataFrame] | None = None
    """
    Data store
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
        if self.data is None:
            raise TypeError

        return self.data[str(data_file)]

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
        if self.data is None:
            self.data = {}

        self.data[str(data_file)] = data
        # Have to do this as, even though it's in-memory,
        # the layer above expects to have files to check, remove etc.
        data_file.touch()


@define
class InMemoryIndexBackend:
    """
    In-memory index backend
    """

    ext: str = ".in-mem"
    """
    Extension to use with files saved by this backend.
    """

    index: pd.DataFrame | None = None
    """Index store"""

    file_map: pd.Series[Path] | None = None  # type: ignore # pandas confused about what it supports
    """File map store"""

    @property
    def preserves_index(self) -> Literal[True]:
        """
        Whether this backend preserves the `pd.MultiIndex` upon (de-)serialisation
        """
        return True

    def load_file_map(self, file_map_file: Path) -> pd.DataFrame:
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
        if self.file_map is None:
            raise TypeError

        return self.file_map.to_frame()

    def load_index(self, index_file: Path) -> pd.DataFrame:
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
        if self.index is None:
            raise TypeError

        return self.index

    def save_file_map(
        self,
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
        self.file_map = file_map
        # Have to do this as, even though it's in-memory,
        # the layer above expects to have files to check
        file_map_file.touch()

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
        self.index = index
        # Have to do this as, even though it's in-memory,
        # the layer above expects to have files to check
        index_file.touch()
