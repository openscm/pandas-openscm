"""
Available back-ends

This is just a shortcut/convenience module
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen

from pandas_openscm.db.csv import CSVDataBackend, CSVIndexBackend
from pandas_openscm.db.feather import FeatherDataBackend, FeatherIndexBackend
from pandas_openscm.db.in_memory import InMemoryDataBackend, InMemoryIndexBackend
from pandas_openscm.db.interfaces import OpenSCMDBDataBackend, OpenSCMDBIndexBackend
from pandas_openscm.db.netcdf import netCDFDataBackend, netCDFIndexBackend


@frozen
class DataBackendOptions:
    """A collection of data back-end options"""

    options: tuple[  # type hint doesn't work properly, but ok
        tuple[str, type[OpenSCMDBDataBackend]], ...
    ]
    """
    Options

    The first element of each option is the option's short name.
    The second element is the class that matches that option.
    """

    def get_instance(self, option: str) -> OpenSCMDBDataBackend:
        """
        Get an instance of one of the options

        Parameters
        ----------
        option
            Option for which to get a data back-end instance

        Returns
        -------
        :
            Initialised instance

        Raises
        ------
        KeyError
            The option is not supported
        """
        for short_name, option_cls in self.options:
            if short_name == option:
                return option_cls()

        msg = (
            f"{option=} is not supported. "
            f"Available options: {tuple(v[1] for v in self.options)}"
        )
        raise KeyError(msg)

    def guess_backend(self, data_file_name: str) -> OpenSCMDBDataBackend:
        """
        Guess backend from a file name

        Parameters
        ----------
        data_file_name
            Name of the data file from which to guess the backend

        Returns
        -------
        :
            Guessed backend

        Raises
        ------
        ValueError
            The backend could not be guessed from `data_file_name`
        """
        ext = Path(data_file_name).suffix
        for _, option_cls in self.options:
            option = option_cls()
            if ext == option.ext:
                return option

        known_options_and_extensions = [(v[0], v[1]().ext) for v in self.options]
        msg = (
            f"Could not guess backend from {data_file_name=!r}. "
            "The file's extension does not match any of the available options: "
            f"{known_options_and_extensions=}"
        )
        raise ValueError(msg)


DATA_BACKENDS = DataBackendOptions(
    (  # type: ignore # using class with protocol doesn't work properly
        ("csv", CSVDataBackend),
        ("feather", FeatherDataBackend),
        ("in_memory", InMemoryDataBackend),
        ("netCDF", netCDFDataBackend),
        # Other options to consider:
        #
        # - pretty netCDF, where we try and save the data with dimensions where possible
        #
        # - HDF5: https://pandas.pydata.org/docs/user_guide/io.html#hdf5-pytables
        # - sqllite
    )
)
"""Inbuilt data back-ends"""


@frozen
class IndexBackendOptions:
    """A collection of index back-end options"""

    options: tuple[tuple[str, type[OpenSCMDBIndexBackend]], ...]
    """
    Options

    The first element of each option is the option's short name.
    The second element is the class that matches that option.
    """

    def get_instance(self, option: str) -> OpenSCMDBIndexBackend:
        """
        Get an instance of one of the options

        Parameters
        ----------
        option
            Option for which to get a index back-end instance

        Returns
        -------
        :
            Initialised instance

        Raises
        ------
        KeyError
            The option is not supported
        """
        for short_name, option_cls in self.options:
            if short_name == option:
                return option_cls()

        msg = (
            f"{option=} is not supported. "
            f"Available options: {tuple(v[1] for v in self.options)}"
        )
        raise KeyError(msg)

    def guess_backend(self, index_file_name: str) -> OpenSCMDBIndexBackend:
        """
        Guess backend from a file name

        Parameters
        ----------
        index_file_name
            Name of the index file from which to guess the backend

        Returns
        -------
        :
            Guessed backend

        Raises
        ------
        ValueError
            The backend could not be guessed from `index_file_name`
        """
        ext = Path(index_file_name).suffix
        for _, option_cls in self.options:
            option = option_cls()
            if ext == option.ext:
                return option

        known_options_and_extensions = [(v[0], v[1]().ext) for v in self.options]
        msg = (
            f"Could not guess backend from {index_file_name=!r}. "
            "The file's extension does not match any of the available options: "
            f"{known_options_and_extensions=}"
        )
        raise ValueError(msg)


INDEX_BACKENDS = IndexBackendOptions(
    (  # type: ignore # using class with protocol doesn't work properly
        ("csv", CSVIndexBackend),
        ("feather", FeatherIndexBackend),
        ("in_memory", InMemoryIndexBackend),
        ("netCDF", netCDFIndexBackend),
        # Other options to consider:
        #
        # - HDF5: https://pandas.pydata.org/docs/user_guide/io.html#hdf5-pytables
        # - sqllite
    )
)
"""Inbuilt index back-ends"""
