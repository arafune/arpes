"""Test for decomposition procedure.

The codes are takedn from full-analysis-xps.ipynb
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arpes.analysis.decomposition import pca_along

if TYPE_CHECKING:
    import xarray as xr


def test_decomposition_pca(xps_map: xr.Dataset) -> None:
    """Test for decomposition procedure.

    The codes are takedn from full-analysis-xps.ipynb
    """
    n_components = 5
    data, pca = pca_along(xps_map.spectrum, ["x", "y"], n_components=n_components)
    np.testing.assert_allclose(
        pca.explained_variance_ratio_,
        np.array([0.64899045, 0.11248539, 0.02484952, 0.01145423, 0.00499509]),
        rtol=1e-5,
    )
