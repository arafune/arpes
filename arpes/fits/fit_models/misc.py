"""Some miscellaneous model definitions."""
import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import update_param_vals
from numpy._typing import NDArray

from arpes._typing import NAN_POLICY

from .x_model_mixin import XModelMixin

__all__ = [
    "QuadraticModel",
    "FermiVelocityRenormalizationModel",
    "LogRenormalizationModel",
]


class QuadraticModel(XModelMixin):
    """A model for fitting a quadratic function."""

    @staticmethod
    def quadratic(
        x: NDArray[np.float_],
        a: float = 1,
        b: float = 0,
        c: float = 0,
    ) -> NDArray[np.float_]:
        """Quadratc polynomial."""
        return a * x**2 + b * x + c

    def __init__(
        self,
        independent_vars: list[str] | None = None,
        prefix: str = "",
        nan_policy: NAN_POLICY = "raise",
        **kwargs,
    ) -> None:
        """Just defer to lmfit for initialization."""
        if independent_vars is None:
            independent_vars = ["x"]
        assert isinstance(independent_vars, list)
        kwargs.update(
            {"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars},
        )
        super().__init__(self.quadratic, **kwargs)

    def guess(self, data: xr.DataArray | NDArray[np.float_], x=None, **kwargs):
        """Placeholder for parameter guesses."""
        pars = self.make_params()

        pars["%sa" % self.prefix].set(value=0)
        pars["%sb" % self.prefix].set(value=0)
        pars["%sc" % self.prefix].set(value=data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiVelocityRenormalizationModel(XModelMixin):
    """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials."""

    @staticmethod
    def fermi_velocity_renormalization_mfl(
        x: NDArray[np.float_],
        n0: float,
        v0: float,
        alpha: float,
        eps: float,
    ) -> NDArray[np.float_]:
        """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials.

        Args:
            x: value to evaluate fit at (carrier density)
            n0: Value of carrier density at cutoff energy for validity of Dirac fermions
            v0: Bare velocity
            alpha: Fine structure constant
            eps: Graphene Dielectric constant
        """
        return v0 * (1 + (alpha / (1 + eps * x**2)) * np.log(n0 / np.abs(x)))

    def __init__(
        self,
        independent_vars: list[str] | None = None,
        prefix: str = "",
        nan_policy: NAN_POLICY = "raise",
        **kwargs,
    ) -> None:
        """Sets physically reasonable constraints on parameter values."""
        if independent_vars is None:
            independent_vars = ["x"]
        assert isinstance(independent_vars, list)
        kwargs.update(
            {"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars},
        )
        super().__init__(self.fermi_velocity_renormalization_mfl, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("n0", min=0.0)
        self.set_param_hint("eps", min=0.0)

    def guess(self, data, x=None, **kwargs) -> lf.Parameters:
        """Placeholder for parameter estimation."""
        pars = self.make_params()

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class LogRenormalizationModel(XModelMixin):
    """A model for Logarithmic Renormalization to Linear Dispersion in Dirac Materials."""

    @staticmethod
    def log_renormalization(  # noqa: PLR0913
        x: NDArray[np.float_],
        kF: float = 1.6,  # noqa:  N803
        kD: float = 1.6,  # noqa:  N803
        kC: float = 1.7,  # noqa:  N803
        alpha: float = 0.4,
        vF: float = 1e6,  # noqa:  N803
    ) -> NDArray[np.float_]:
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

    def __init__(
        self,
        independent_vars: list[str] | None = None,
        prefix: str = "",
        nan_policy: NAN_POLICY = "raise",
        **kwargs,
    ) -> None:
        """Initialize.

        The fine structure constant and velocity must be positive, so we will constrain them here.
        """
        if independent_vars is None:
            independent_vars = ["x"]
        assert isinstance(independent_vars, list)
        kwargs.update(
            {"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars},
        )
        super().__init__(self.log_renormalization, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("vF", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for actually making parameter estimates here."""
        pars = self.make_params()

        pars["%skC" % self.prefix].set(value=1.7)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
