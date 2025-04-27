"""
Tools that support comparisons between [pd.DataFrame][pandas.DataFrame]'s
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas_openscm.typing import NP_ARRAY_OF_BOOL, NP_ARRAY_OF_FLOAT_OR_INT


def compare_close(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    isclose: Callable[
        [NP_ARRAY_OF_FLOAT_OR_INT, NP_ARRAY_OF_FLOAT_OR_INT], NP_ARRAY_OF_BOOL
    ] = np.isclose,
) -> pd.DataFrame:
    """
    Compare two [pd.DataFrame][pandas.DataFrame]'s

    This is like [pd.DataFrame.compare][pandas.DataFrame.compare]
    except you can specify the function to determine
    whether values are close or not.

    Parameters
    ----------
    left
        First [pd.DataFrame][pandas.DataFrame] to compare

    right
        Other [pd.DataFrame][pandas.DataFrame] to compare

    left_name
        Name of `left` to use in the result

    right_name
        Name of `right` to use in the result

    isclose
        Function to use to determine whether values are close

        (Hint: use [functools.partial][] to specify a custom
        tolerance with [np.isclose][numpy.isclose].)

    Returns
    -------
    :
        The comparison between `left` and `right` at the provided tolerance

        Only indexes where `left` and `right` differ are returned,
        i.e. if the result is empty, `left` and `right` are equal for all indexes.
    """
    left_stacked = left.stack()
    left_stacked.name = left_name

    right_stacked = right.stack()
    right_stacked.name = right_name

    left_stacked_aligned, right_stacked_aligned = left_stacked.align(right_stacked)
    differences_locator = ~isclose(
        left_stacked_aligned.values, right_stacked_aligned.values
    )

    res = pd.concat(
        [
            left_stacked_aligned[differences_locator],
            right_stacked_aligned[differences_locator],
        ],
        axis="columns",
    )

    return res
