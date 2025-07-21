import numpy as np
import pytest
from lmfit import Parameters

from arpes.fits import (
    AffineBroadenedFD,
    FermiDiracModel,
    FermiLorentzianModel,
    GStepBModel,
)


@pytest.mark.parametrize(
    "ModelClass, params",
    [
        (
            AffineBroadenedFD,
            {"center": 0, "width": 0.2, "const_bkg": 0.1, "lin_slope": 0.01, "sigma": 0.1},
        ),
        # (BandEdgeBGModel, {"center": 0, "width": 0.2, "a": 0.1, "b": 0.01}),
        # (BandEdgeBModel, {"center": 0, "width": 0.2, "b": 0.01}),
        (FermiDiracModel, {"center": 0, "width": 0.2}),
        (FermiLorentzianModel, {"center": 0, "width": 0.2, "amplitude": 1, "sigma": 0.1}),
        (GStepBModel, {"center": 0, "width": 0.2, "amplitude": 1, "b": 0.01, "sigma": 0.1}),
        # (GStepBStandardModel, {"center": 0, "width": 0.2, "amplitude": 1, "b": 0.01, "sigma": 0.1}),
    ],
)
def test_model_evaluation(ModelClass, params):
    model = ModelClass(prefix="")  # prefixによるパラメータ名混乱を防ぐ
    x = np.linspace(-1, 1, 201)
    param_obj = Parameters()
    for name, val in params.items():
        param_obj.add(name, value=val)
    y = model.eval(params=param_obj, x=x)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
