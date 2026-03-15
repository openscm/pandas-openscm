"""
Unit tests of [pandas_openscm.plotting][]
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from pandas_openscm.plotting.axis_labels import (
    handle_axis_label_inference_from_unit_information,
)
from pandas_openscm.testing import create_test_df


@pytest.mark.parametrize(
    [
        "label",
        "unit_aware",
        "pandas_obj",
        "unit_index_level",
        "warn_infer_label_with_multi_unit",
        "exp",
        "exp_warn_error",
    ],
    (
        (
            "custom_string",
            "not_used",
            "not_used",
            "not_used",
            "not_used",
            "custom_string",
            does_not_raise(),
        ),
        (
            None,
            "not_used",
            "not_used",
            "not_used",
            "not_used",
            None,
            does_not_raise(),
        ),
        (
            False,
            "not_used",
            "not_used",
            "not_used",
            "not_used",
            None,
            does_not_raise(),
        ),
        (
            True,
            "not_used",
            "not_used",
            None,
            "not_used",
            None,
            does_not_raise(),
        ),
        (
            ["list", "of", "values"],
            "not_used",
            "not_used",
            "variable",
            "not_used",
            None,
            pytest.raises(
                TypeError,
                match=re.escape(
                    "<class 'list'> are not supported. label=['list', 'of', 'values']"
                ),
            ),
        ),
        (
            True,
            True,
            "not_used",
            "not_used",
            "not_used",
            None,
            does_not_raise(),
        ),
        (
            True,
            False,
            create_test_df(
                variables=(("Warming", "K"),),
                n_scenarios=5,
                n_runs=10,
                timepoints=np.arange(1950.0, 1965.0),
            ),
            "units",
            "not_used",
            None,
            pytest.warns(
                UserWarning,
                match="Not auto-generating the label "
                "because unit_index_level='units' is not in pandas_obj.index.names=",
            ),
        ),
        (
            True,
            False,
            create_test_df(
                variables=(("Warming", "K"),),
                n_scenarios=5,
                n_runs=10,
                timepoints=np.arange(1950.0, 1965.0),
            ),
            "unit",
            "not_used",
            "K",
            does_not_raise(),
        ),
        (
            True,
            False,
            create_test_df(
                variables=(("Warming", "K"), ("Emissions", "tC / yr")),
                n_scenarios=5,
                n_runs=10,
                timepoints=np.arange(1950.0, 1965.0),
            ),
            "unit",
            True,
            None,
            pytest.warns(
                UserWarning,
                match=re.escape(
                    "Not auto-generating the label "
                    "because the data has more than one unit: "
                    "data units ['K', 'tC / yr']"
                ),
            ),
        ),
        (
            True,
            False,
            create_test_df(
                variables=(("Warming", "K"), ("Emissions", "tC / yr")),
                n_scenarios=5,
                n_runs=10,
                timepoints=np.arange(1950.0, 1965.0),
            ),
            "unit",
            False,
            None,
            does_not_raise(),
        ),
    ),
)
def test_handle_axis_label_inference_from_unit_information(  # noqa: PLR0913
    label,
    unit_aware,
    pandas_obj,
    unit_index_level,
    warn_infer_label_with_multi_unit,
    exp,
    exp_warn_error,
):
    with exp_warn_error:
        res = handle_axis_label_inference_from_unit_information(
            label=label,
            unit_aware=unit_aware,
            pandas_obj=pandas_obj,
            unit_index_level=unit_index_level,
            warn_infer_label_with_multi_unit=warn_infer_label_with_multi_unit,
        )

    if isinstance(exp_warn_error, does_not_raise):
        assert res == exp
