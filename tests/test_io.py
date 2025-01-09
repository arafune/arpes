"""Unit test of io module in aryspes."""

import pytest

from arpes.io import load_example_data


def test_load_example_raises_kye_error() -> None:
    msg = "Could not find requested example_name: cut0.*"
    with pytest.raises(KeyError, match=msg):
        load_example_data("cut0")
