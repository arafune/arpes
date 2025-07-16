"""Unit test for analysis.xps."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arpes.analysis.xps import approximate_core_levels
import arpes.xarray_extensions


def test_approximate_core_levels(xps_map: xr.Dataset) -> None:
    """Test the core level approximation function."""
    xps_spectrum = xps_map.spectrum.sum(["x", "y"], keep_attrs=True)
    approx_levels = approximate_core_levels(xps_spectrum, promenance=5)
    np.testing.assert_almost_equal(approx_levels[0], -34.5066501491)
