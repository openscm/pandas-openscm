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

    Examples
    --------
    >>> import pandas as pd
    >>> left = pd.DataFrame(
    ...     [[1.0, 2.0, 3.0], [1.1, 1.2, 1.3], [-1.1, 0.0, 0.5]],
    ...     columns=pd.Index([2.0, 4.0, 10.0], name="time"),
    ...     index=pd.MultiIndex.from_tuples(
    ...         [("v1", "kg"), ("v2", "m"), ("v3", "yr")], names=["variable", "unit"]
    ...     ),
    ... )
    >>> left
    time           2.0   4.0   10.0
    variable unit
    v1       kg     1.0   2.0   3.0
    v2       m      1.1   1.2   1.3
    v3       yr    -1.1   0.0   0.5
    >>>
    >>> right = pd.DataFrame(
    ...     [[1.1, 2.1, 3.1], [1.11, 1.2, 1.31], [-1.12, 0.0000001, 0.5]],
    ...     columns=pd.Index([2.0, 4.0, 10.0], name="time"),
    ...     index=pd.MultiIndex.from_tuples(
    ...         [("v1", "kg"), ("v2", "m"), ("v3", "yr")], names=["variable", "unit"]
    ...     ),
    ... )
    >>> right
    time           2.0           4.0   10.0
    variable unit
    v1       kg    1.10  2.100000e+00  3.10
    v2       m     1.11  1.200000e+00  1.31
    v3       yr   -1.12  1.000000e-07  0.50

    >>>
    >>> # Default tolerances are quite tight
    >>> compare_close(left, right, "left", "right")
                        left         right
    variable unit time
    v1       kg   2.0    1.0  1.100000e+00
                  4.0    2.0  2.100000e+00
                  10.0   3.0  3.100000e+00
    v2       m    2.0    1.1  1.110000e+00
                  10.0   1.3  1.310000e+00
    v3       yr   2.0   -1.1 -1.120000e+00
                  4.0    0.0  1.000000e-07
    >>>
    >>> from functools import partial
    >>> import numpy as np
    >>>
    >>> # We can use `functools.partial` to loosen the tolerances
    >>> compare_close(
    ...     left, right, "left", "right", isclose=partial(np.isclose, atol=0.01)
    ... )
                        left  right
    variable unit time
    v1       kg   2.0    1.0   1.10
                  4.0    2.0   2.10
                  10.0   3.0   3.10
    v3       yr   2.0   -1.1  -1.12
    >>>
    >>> compare_close(
    ...     left,
    ...     right,
    ...     # Note you can also change the displayed names
    ...     left_name="Bill",
    ...     right_name="Ben",
    ...     isclose=partial(np.isclose, rtol=0.1),
    ... )
                             Bill           Ben
    variable unit time
    v3       yr   4.0         0.0  1.000000e-07
    >>>
    >>> # If we make the tolerance sufficiently loose,
    >>> # all points are considered equal
    >>> # and the result is empty.
    >>> loose_comparison = compare_close(
    ...     left,
    ...     right,
    ...     "left",
    ...     "right",
    ...     isclose=partial(np.isclose, rtol=0.1, atol=0.001),
    ... )
    >>> loose_comparison.empty
    True
    """
    left_stacked = left.stack()
    left_stacked.name = left_name

    right_stacked = right.stack()
    right_stacked.name = right_name

    left_stacked_aligned, right_stacked_aligned = left_stacked.align(right_stacked)
    differences_locator = ~isclose(
        left_stacked_aligned.values,  # type: ignore
        right_stacked_aligned.values,  # type: ignore
    )

    res = pd.concat(
        [
            left_stacked_aligned[differences_locator],
            right_stacked_aligned[differences_locator],
        ],
        axis="columns",
    )

    return res
