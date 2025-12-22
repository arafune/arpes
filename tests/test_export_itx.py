"""Unit test for export/itx.py."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from arpes.export.itx import convert_itx_format


def test_convert_itx_format(dataarray_cut: xr.DataArray) -> None:
    """Test convert_itx_format."""
    list_style = convert_itx_format(dataarray_cut, add_notes=True).split("\n")
    # assert prodigy_itx.pixels == (240, 240)
