"""Wraps standard lmfit models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
import numpy as np
from lmfit.models import guess_from_peak, update_param_vals

from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes.fits import ModelARGS

__all__ = [
    "VoigtModel",
    "GaussianModel",
    "ConstantModel",
    "LorentzianModel",
    "SkewedVoigtModel",
    "SplitLorentzianModel",
    "LinearModel",
    "LogisticModel",
    "StepModel",
    "SineModel",
]


class SineModel(XModelMixin, lf.models.SineModel):
    """Wraps `lf.models.SineModel`."""


class VoigtModel(XModelMixin, lf.models.VoigtModel):
    """Wraps `lf.models.VoigtModel`."""


class GaussianModel(XModelMixin, lf.models.GaussianModel):
    """Wraps `lf.models.GaussianModel`."""


class ConstantModel(XModelMixin, lf.models.ConstantModel):
    """Wraps `lf.models.ConstantModel`."""


class LorentzianModel(XModelMixin, lf.models.LorentzianModel):
    """Wraps `lf.models.LorentzianModel`."""


class SkewedVoigtModel(XModelMixin, lf.models.SkewedVoigtModel):
    """Wraps `lf.models.SkewedVoigtModel`."""


class SkewedGaussianModel(XModelMixin, lf.models.SkewedGaussianModel):
    """Wraps `lf.models.SkewedGaussianModel`."""


class SplitLorentzianModel(XModelMixin, lf.models.SplitLorentzianModel):
    """Wraps `lf.models.SplitLorentzianModel`."""

    def guess(
        self,
        data: xr.DataArray | NDArray[np.float_],
        x: NDArray[np.float_] | None = None,
        **kwargs: float,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        pars = self.make_params()
        pars = guess_from_peak(self, data, x, negative=False, ampscale=1.25)
        sigma = pars["%ssigma" % self.prefix]
        pars["%ssigma_r" % self.prefix].set(value=sigma.value, min=sigma.min, max=sigma.max)

        return update_param_vals(pars, self.prefix, **kwargs)


class LinearModel(XModelMixin, lf.models.LinearModel):
    """A linear regression model."""

    def guess(
        self,
        data: xr.DataArray | NDArray[np.float_],
        x: NDArray[np.float_] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Use np.polyfit to get good initial parameters."""
        sval, oval = 0.0, 0.0
        if x is not None:
            sval, oval = np.polyfit(x, data, 1)
        pars = self.make_params(intercept=oval, slope=sval)
        return update_param_vals(pars, self.prefix, **kwargs)


class LogisticModel(XModelMixin, lf.models.StepModel):
    """A logistic regression model."""

    def __init__(self, **kwargs: Unpack[ModelARGS]) -> None:
        """Set standard parameters and delegate to lmfit."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        kwargs.update(
            {
                "form": "logistic",
            },
        )
        super().__init__(**kwargs)


class StepModel(XModelMixin, lf.models.StepModel):
    """Wraps `lf.models.StepModel`."""
