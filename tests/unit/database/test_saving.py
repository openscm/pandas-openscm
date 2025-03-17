"""
Tests of `pandas_openscm.db.saving`
"""

from __future__ import annotations

import pytest

from pandas_openscm.db.saving import SaveAction, save_file


def test_save_file_unrecognised_save_action_info_kind_error():
    info_kind = 12

    with pytest.raises(NotImplementedError, match=str(info_kind)):
        save_file(
            SaveAction(
                info="not used",
                info_kind=info_kind,
                backend="not used",
                save_path="not used",
            )
        )
