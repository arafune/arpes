"""Definitions of models involving Dirac points, graphene, graphite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
import xarray as xr
from lmfit.lineshapes import lorentzian
from lmfit.models import Model, update_param_vals

from arpes._typing import XrTypes

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from arpes.fits import ModelArgs

__all__ = ("DiracDispersionModel",)


class DiracDispersionModel(Model):
    """Model for dirac_dispersion symmetric about the dirac point."""

    def dirac_dispersion(  # noqa:  PLR0913
        self,
        x: NDArray[np.float64],
        kd: float = 1.6,
        amplitude_1: float = 1,
        amplitude_2: float = 1,
        center: float = 0,
        sigma_1: float = 1,
        sigma_2: float = 1,
    ) -> NDArray[np.float64]:
        """Model for dirac_dispersion symmetric about the dirac point.

        Fits lorentziants to (kd-center) and (kd+center)

        Args:
            x: value to evaluate fit at
            kd: Dirac point momentum
            amplitude_1: amplitude of Lorentzian at kd-center
            amplitude_2: amplitude of Lorentzian at kd+center
            center: center of Lorentzian
            sigma_1: FWHM of Lorentzian at kd-center
            sigma_2: FWHM of Lorentzian at kd+center

        Returns:
            An MDC model for a Dirac like dispersion around the cone.
        """
        return lorentzian(
            x,
            center=kd - center,
            amplitude=amplitude_1,
            sigma=sigma_1,
        ) + lorentzian(
            x,
            center=kd + center,
            amplitude=amplitude_2,
            sigma=sigma_2,
        )

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.dirac_dispersion, **kwargs)

        self.set_param_hint("sigma_1", min=0.0)
        self.set_param_hint("sigma_2", min=0.0)

    def guess(
        self,
        data: NDArray[np.float64] | XrTypes,
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Estimate initial model parameter values from data."""
        if isinstance(data, XrTypes):
            data = data.values
        if isinstance(x, xr.DataArray):
            x = x.values
        pars = self.make_params()
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = (
        "Model for dirac_dispersion symmetric about the dirac point." + lf.models.COMMON_INIT_DOC
    )
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
