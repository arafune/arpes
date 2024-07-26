"""Unit test for utility/combine.py/concat_along_phi."""

import xarray as xr
from arpes.utilities import concat_along_phi


def test_concat_along_phi(mote2_1: xr.Dataset, mote2_2: xr.Dataset):
    """Test for concat_along_phi.

    Todo:
        The test would not be sufficient, add more.

    Args:
        mote2_1: Dataset of MoTe2
        mote2_2: Dataset of MoTe2
    """
    mote2_1_array = mote2_1.spectrum.S.corrected_angle_by("beta")
    mote2_2_array = mote2_2.spectrum.S.corrected_angle_by("beta")
    concat_mote2 = concat_along_phi(mote2_1_array, mote2_2_array)
    assert concat_mote2.attrs["provenance"]["parent_id"] == [49, 54]
    concat_mote2_ = concat_along_phi(
        mote2_1_array,
        mote2_2_array,
        enhance_a=1 / 1.54,
        occupation_ratio=0.58,
    )
    assert concat_mote2_.attrs["provenance"]["parent_id"] == [49, 54]
