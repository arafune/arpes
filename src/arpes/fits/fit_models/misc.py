"""Some miscellaneous model definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import Model, update_param_vals

from arpes._typing.base import XrTypes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arpes._typing.fits import ModelArgs

__all__ = (
    "FermiVelocityRenormalizationModel",
    "LogRenormalizationModel",
)


class FermiVelocityRenormalizationModel(Model):
    """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials."""

    @staticmethod
    def fermi_velocity_renormalization_mfl(
        x: NDArray[np.float64],
        n0: float,
        v0: float,
        alpha: float,
        eps: float,
    ) -> NDArray[np.float64]:
        """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials.

        Args:
            x: value to evaluate fit at (carrier density)
            n0: Value of carrier density at cutoff energy for validity of Dirac fermions
            v0: Bare velocity
            alpha: Fine structure constant
            eps: Graphene Dielectric constant
        """
        return v0 * (1 + (alpha / (1 + eps * x**2)) * np.log(n0 / np.abs(x)))

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Sets physically reasonable constraints on parameter values."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.fermi_velocity_renormalization_mfl, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("n0", min=0.0)
        self.set_param_hint("eps", min=0.0)

    def guess(
        self,
        data: XrTypes | NDArray[np.float64],
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for parameter estimation."""
        if isinstance(x, xr.DataArray):
            x = x.values
        if isinstance(data, XrTypes):
            data = np.asarray(data.values)
        pars = self.make_params()

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = "Fermi velocity renormalization model" + lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class LogRenormalizationModel(Model):
    """A model for Logarithmic Renormalization to Linear Dispersion in Dirac Materials."""

    @staticmethod
    def log_renormalization(  # noqa: PLR0913
        x: NDArray[np.float64],
        kF: float = 1.6,  # noqa:  N803
        kD: float = 1.6,  # noqa:  N803
        kC: float = 1.7,  # noqa:  N803
        alpha: float = 0.4,
        vF: float = 1e6,  # noqa:  N803
    ) -> NDArray[np.float64]:
        """Logarithmic correction to linear dispersion near charge neutrality in Dirac materials.

        As examples, this can be used to study the low energy physics in high quality ARPES spectra
        of graphene or topological Dirac semimetals.

        Args:
            x: The coorindates for the fit
            k: value to evaluate fit at
            kF: Fermi wavevector
            kD: Dirac point
            alpha: Fine structure constant
            vF: Bare Band Fermi Velocity
            kC: Cutoff Momentum
        """
        dk = x - kF
        dkD = x - kD
        return -vF * np.abs(dkD) + (alpha / 4) * vF * dk * np.log(np.abs(kC / dkD))

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Initialize.

        The fine structure constant and velocity must be positive, so we will constrain them here.
        """
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(self.log_renormalization, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("vF", min=0.0)

    def guess(
        self,
        data: XrTypes | NDArray[np.float64],
        x: NDArray[np.float64] | xr.DataArray,
        **kwargs: float,
    ) -> lf.Parameters:
        """Placeholder for actually making parameter estimates here."""
        if isinstance(data, XrTypes):
            data = np.asarray(data.values)
        if isinstance(x, xr.DataArray):
            x = x.values
        pars = self.make_params()

        pars[f"{self.prefix}kC"].set(value=1.7)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = "Log renormalization model" + lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
